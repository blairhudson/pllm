//! Shared, allocation-free tensor contracts. These are internal interpretations
//! of existing operation attributes, not a new serialized schema or authority.
use crate::ModelOperation;
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingDomain { Tokens, AbsolutePositions, Segments }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingError { Source, Contract }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Embedding {
    pub domain: EmbeddingDomain,
    /// None only for the original input.tokens lookup with no declared shape.
    /// Requirements may resolve that legacy case from the phase vocabulary.
    pub cardinality: Option<u64>,
}

pub(crate) fn declared_shape(value: Option<&Value>, dimensions: &[u64]) -> bool {
    value.and_then(Value::as_array).is_some_and(|shape| {
        shape.len() == dimensions.len() && shape.iter().zip(dimensions)
            .all(|(value, expected)| value.as_u64() == Some(*expected))
    })
}

pub(crate) fn projection_dimensions(op: &ModelOperation, input: u64, output: u64)
    -> Option<[u64; 2]> {
    projection_dimensions_parts(&op.attributes,input,output)
}

pub(crate) fn projection_dimensions_parts(attributes: &Value, input: u64, output: u64)
    -> Option<[u64;2]> {
    let dimensions = match attributes.get("weight_layout") {
        None => [output, input],
        Some(Value::String(layout)) if layout == "output_input" => [output, input],
        Some(Value::String(layout)) if layout == "input_output" => [input, output],
        _ => return None,
    };
    if attributes.get("weight_shape").is_some()
        && !declared_shape(attributes.get("weight_shape"), &dimensions)
        || attributes.get("bias_shape").is_some()
            && !declared_shape(attributes.get("bias_shape"), &[output])
    {
        return None;
    }
    Some(dimensions)
}

/// Describes an existing TokenLookup's domain and declared table cardinality.
/// Callers must first classify the operator as TokenLookup. This predicate does
/// not authenticate a source, validate a graph, or authorize local execution.
pub fn embedding(op: &ModelOperation) -> Result<Embedding, EmbeddingError> {
    let width = op.output_shape.last().copied().unwrap_or(0);
    embedding_parts(&op.inputs,&op.attributes,width)
}

pub(crate) fn embedding_parts(inputs: &[String], attributes: &Value, width: u64) -> Result<Embedding,EmbeddingError> {
    use EmbeddingDomain::*;
    use EmbeddingError::*;
    if inputs.len() != 1 { return Err(Source); }
    let domain = match (inputs[0].as_str(), attributes.get("index_domain")) {
        ("input.tokens", None) => Tokens,
        ("input.tokens", Some(Value::String(s))) if s == "tokens" => Tokens,
        ("input.positions", Some(Value::String(s))) if s == "absolute_positions" => AbsolutePositions,
        ("input.segments", Some(Value::String(s))) if s == "segments" => Segments,
        _ => return Err(Source),
    };
    if width == 0 { return Err(Contract); }
    let cardinality = match attributes.get("weight_shape") {
        None if domain == Tokens && attributes.get("index_domain").is_none() => None,
        Some(Value::Array(shape)) if shape.len() == 2 => {
            let count = shape[0].as_u64().filter(|n| *n > 0).ok_or(Contract)?;
            if shape[1].as_u64() != Some(width) { return Err(Contract); }
            Some(count)
        }
        _ => return Err(Contract),
    };
    if attributes.get("weight_layout").is_some_and(|v| v.as_str() != Some("output_input")) {
        return Err(Contract);
    }
    Ok(Embedding { domain, cardinality })
}

/// Parse directly to native FP32, preserving Rust's existing decimal grammar
/// and rounding. Never parse through FP64: double rounding changes accepted
/// epsilon values near FP32 midpoints. This pure scalar helper grants no
/// source, placement, or execution authority.
pub fn parse_positive_f32(text: &str) -> Result<f32, &'static str> {
    let value = text.parse::<f32>().map_err(|_| "invalid positive FP32 decimal")?;
    if !value.is_finite() || value <= 0.0 {
        return Err("FP32 value must be finite and positive");
    }
    Ok(value)
}

/// Stateless key-padding mask, applied additively to FP32 scores. Runtime must
/// additionally check a bool [batch, keys] mask with at least one visible key in
/// each batch. No causal, query-padding, or sequential-position semantics.
pub fn key_padding_attention_mask(inputs: &[String], attributes: &Value) -> bool {
    let Some(a) = attributes.as_object() else { return false; };
    inputs.len() == 2 && inputs[1] == "input.attention_mask" && a.len() == 6
        && a.get("kind").and_then(Value::as_str) == Some("key_padding")
        && a.get("mask_input").and_then(Value::as_str) == Some("input.attention_mask")
        && a.get("broadcast").and_then(Value::as_str) == Some("batch_heads_queries_keys")
        && a.get("masked_value").and_then(Value::as_str) == Some("float32_min")
        && a.get("application").and_then(Value::as_str) == Some("additive")
        && a.get("compute_dtype").and_then(Value::as_str) == Some("float32")
}

pub fn exact_fp32_gelu(attributes: &Value) -> bool {
    attributes.as_object().is_some_and(|a| a.len() == 2
        && a.get("formula").and_then(Value::as_str) == Some("x/2*(1+erf(x/sqrt(2)))")
        && a.get("compute_dtype").and_then(Value::as_str) == Some("float32"))
}

pub fn fp32_tanh(attributes: &Value) -> bool {
    attributes.as_object().is_some_and(|a| a.len() == 1
        && a.get("compute_dtype").and_then(Value::as_str) == Some("float32"))
}

/// Existing centered, population-variance affine FP32 LayerNorm contract.
/// Shape equality with the source is checked by graph validation, not here.
/// This descriptive predicate grants no placement or execution authority.
pub fn centered_affine_layer_norm(op: &ModelOperation) -> bool {
    let Some(width) = op.output_shape.last().copied().filter(|n| *n > 0) else { return false; };
    centered_affine_layer_norm_parts(op.inputs.len(),&op.attributes,width)
}

pub(crate) fn centered_affine_layer_norm_parts(input_count: usize, a: &Value, width: u64) -> bool {
    let eps = a.get("epsilon").and_then(Value::as_str).and_then(|v| parse_positive_f32(v).ok());
    width > 0 && input_count == 1
        && a.get("axis").and_then(Value::as_i64) == Some(-1)
        && a.get("variance").and_then(Value::as_str) == Some("centered_mean_square")
        && a.get("compute_dtype").and_then(Value::as_str) == Some("float32")
        && a.get("output_dtype").is_none_or(|v| v.as_str() == Some("float32"))
        && a.get("epsilon_placement").and_then(Value::as_str) == Some("inside_sqrt")
        && a.get("affine_order").and_then(Value::as_str) == Some("normalize_multiply_add")
        && eps.is_some()
        && a.get("weight").and_then(Value::as_str).is_some_and(|v| !v.is_empty())
        && a.get("bias").and_then(Value::as_str).is_some_and(|v| !v.is_empty())
        && declared_shape(a.get("weight_shape"), &[width])
        && declared_shape(a.get("bias_shape"), &[width])
}

pub(crate) fn validate_layer_norm(
    op: &ModelOperation,
    shapes: &std::collections::BTreeMap<String, Vec<u64>>,
) -> Result<(), crate::ModelError> {
    if !centered_affine_layer_norm(op)
        || op.inputs.first().and_then(|input| shapes.get(input)) != Some(&op.output_shape)
    {
        return Err(crate::ModelError::Incomplete("invalid centered affine LayerNorm contract".into()));
    }
    Ok(())
}
