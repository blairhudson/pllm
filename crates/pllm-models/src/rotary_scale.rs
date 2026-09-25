//! Bounded, model-neutral rotary frequency scaling descriptors.

use super::ModelError;
use serde_json::{json, Value};

pub(crate) const WAVELENGTH_COEFFICIENT_PROFILE: &str = "pllm.numeric.rope.float32.wavelength.v1";
pub(crate) const PER_FREQUENCY_COEFFICIENT_PROFILE: &str =
    "pllm.numeric.rope.float32.per_frequency.v1";
const MAX_POSITION: u64 = 1 << 24;

pub(crate) fn source_descriptor(
    source: Option<&Value>,
    max_position_embeddings: u64,
) -> Result<Option<Value>, ModelError> {
    let Some(source) = source else {
        return Ok(None);
    };
    let fields = source.as_object().ok_or_else(|| {
        ModelError::Unsupported("rotary frequency scaling must be an object".into())
    })?;
    if fields.len() != 5 || fields.get("rope_type").and_then(Value::as_str) != Some("llama3") {
        return Err(ModelError::Unsupported(
            "only bounded llama3 wavelength-transition RoPE is supported".into(),
        ));
    }
    let descriptor = json!({
        "kind": "wavelength_transition",
        "factor": fields.get("factor"),
        "low_freq_factor": fields.get("low_freq_factor"),
        "high_freq_factor": fields.get("high_freq_factor"),
        "original_max_position_embeddings": fields.get("original_max_position_embeddings"),
    });
    if !valid_wavelength_descriptor(&descriptor)
        || max_position_embeddings > MAX_POSITION
        || descriptor["original_max_position_embeddings"]
            .as_u64()
            .is_none_or(|original| original >= max_position_embeddings)
    {
        return Err(ModelError::Unsupported(
            "rotary wavelength transition exceeds its numeric or context bounds".into(),
        ));
    }
    Ok(Some(descriptor))
}

pub(crate) fn valid_wavelength_descriptor(value: &Value) -> bool {
    let Some(fields) = value.as_object() else {
        return false;
    };
    if fields.len() != 5
        || fields.get("kind").and_then(Value::as_str) != Some("wavelength_transition")
    {
        return false;
    }
    let (Some(factor), Some(low), Some(high), Some(original)) = (
        fields.get("factor").and_then(Value::as_f64),
        fields.get("low_freq_factor").and_then(Value::as_f64),
        fields.get("high_freq_factor").and_then(Value::as_f64),
        fields
            .get("original_max_position_embeddings")
            .and_then(Value::as_u64),
    ) else {
        return false;
    };
    factor.is_finite()
        && (1.0..=256.0).contains(&factor)
        && (factor as f32) > 1.0
        && low.is_finite()
        && (1.0 / 256.0..=256.0).contains(&low)
        && high.is_finite()
        && (high as f32) - (low as f32) >= 1.0 / 256.0
        && high <= 256.0
        && (1..=MAX_POSITION).contains(&original)
}

pub(crate) fn valid_per_frequency_descriptor(value: &Value, rotary_dimensions: u64) -> bool {
    let Some(fields) = value.as_object() else {
        return false;
    };
    if fields.len() != 5
        || fields.get("kind").and_then(Value::as_str) != Some("per_frequency_context")
        || rotary_dimensions == 0
        || rotary_dimensions % 2 != 0
    {
        return false;
    }
    let (Some(original), Some(factor), Some(short), Some(long)) = (
        fields
            .get("original_max_position_embeddings")
            .and_then(Value::as_u64),
        fields.get("factor").and_then(Value::as_f64),
        fields.get("short_factor").and_then(Value::as_array),
        fields.get("long_factor").and_then(Value::as_array),
    ) else {
        return false;
    };
    let width = usize::try_from(rotary_dimensions / 2).ok();
    (2..=MAX_POSITION).contains(&original)
        && factor.is_finite()
        && (1.0..=256.0).contains(&factor)
        && width.is_some_and(|width| short.len() == width && long.len() == width)
        && short.iter().chain(long).all(|value| {
            value
                .as_f64()
                .is_some_and(|factor| factor.is_finite() && (1.0 / 256.0..=256.0).contains(&factor))
        })
}
