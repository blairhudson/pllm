//! Caller-selected bounds for source expansion before semantic graph allocation.
//! Source mappings opt in only when these dimensions bound their graph builder.

use crate::{DecoderPlan, DecoderWorkload, ModelError};
use serde_json::Value;

/// Opt-in metadata bounds; these do not authenticate the source or provide a
/// counter reservation. The compiler frontend must keep its real owner/charge.
#[derive(Clone, Copy, Debug)]
pub struct ConfigLowererLimits {
    pub max_operations_per_phase: usize,
    pub max_string_bytes: usize,
    pub max_attribute_nodes: usize,
    pub max_retained_strings: usize,
    pub max_factory_bytes: usize,
}

/// A quotation of the actual registered source factory, borrowing its immutable
/// configuration. Cannot be rebound to a different document or workload. It is
/// NOT a source certificate or an allocation permit. Legacy byte lowering is
/// unchanged; an owning frontend reserves before calling `construct`.
pub struct ConfigLowererQuote<'a> {
    document: &'a Value,
    workload: DecoderWorkload,
    lower: fn(&Value, DecoderWorkload) -> Result<DecoderPlan, ModelError>,
    metadata: SourceMetadataEstimate,
    factory_bytes: usize,
}

impl<'a> ConfigLowererQuote<'a> {
    pub fn new(
        document: &'a Value,
        workload: DecoderWorkload,
        expansion: DecoderExpansionBudget,
        limits: ConfigLowererLimits,
    ) -> Result<Self, ModelError> {
        let lower = super::source_mappings::admitted_lowerer(document, workload, expansion)?;
        let metadata = expansion.preflight_metadata(
            document,
            workload,
            limits.max_operations_per_phase,
            limits.max_string_bytes,
            limits.max_attribute_nodes,
            limits.max_retained_strings,
        )?;
        // Real existing document, not caller-reported baseline/capacity. Original
        // source backing is retained/charged by the caller; these are EXTRA copy
        // and canonicalization allowances under the registered closed builders.
        let source_bytes =
            crate::owned_backing::checked_json_backing_bytes(document, limits.max_factory_bytes)
                .map_err(|_| rejected())?;
        let operations = metadata
            .operations_per_phase
            .checked_mul(2)
            .ok_or_else(rejected)?;
        let states = metadata
            .states_per_phase
            .checked_mul(4)
            .ok_or_else(rejected)?;
        // Validation retains available IDs/shapes and input/output state indexes.
        // Reuse the shared opaque-node policy, not an assumed private std layout.
        let validation_entries = operations
            .checked_mul(4)
            .and_then(|n| states.checked_mul(4).and_then(|s| n.checked_add(s)))
            .and_then(|n| n.checked_add(32))
            .ok_or_else(rejected)?;
        let node = crate::owned_backing::checked_tree_node_bytes::<(String, Vec<u64>)>()
            .map_err(|_| rejected())?;
        let validation = validation_entries.checked_mul(node).ok_or_else(rejected)?;
        // Both phase results overlap. An additional full builder envelope covers
        // reallocating Vec/string backing while its prior allocation still lives;
        // four source envelopes cover typed config + Value/canonical temporaries.
        // This is logical owner admission, not allocator overhead/stack/RSS.
        let factory_bytes = metadata
            .retained_metadata_bytes
            .checked_mul(2)
            .and_then(|n| source_bytes.checked_mul(4).and_then(|s| n.checked_add(s)))
            .and_then(|n| n.checked_add(validation))
            .and_then(|n| n.checked_add(std::mem::size_of::<DecoderPlan>()))
            .ok_or_else(rejected)?;
        if factory_bytes > limits.max_factory_bytes {
            return Err(rejected());
        }
        Ok(Self {
            document,
            workload,
            lower,
            metadata,
            factory_bytes,
        })
    }

    pub fn metadata(&self) -> SourceMetadataEstimate {
        self.metadata
    }
    pub fn factory_bytes(&self) -> usize {
        self.factory_bytes
    }

    /// Must run inside the frontend's already retained real reservation. This
    /// invokes the original semantic factory; no serialized plan/DTO surrogate.
    pub fn construct(self) -> Result<DecoderPlan, ModelError> {
        #[cfg(test)]
        FACTORY_ATTEMPTS.with(|n| n.set(n.get() + 1));
        (self.lower)(self.document, self.workload)
    }
}

#[cfg(test)]
thread_local! {
    static FACTORY_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn factory_attempts() -> usize {
    FACTORY_ATTEMPTS.with(|n| n.get())
}

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
        let operations = checked_phase_operations(layers, per_layer, fixed_operations)?;
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
                operations
                    .checked_mul(2)
                    .and_then(|o| {
                        o.checked_mul(std::mem::size_of::<crate::ModelOperation>() + 512 + 8192)
                    })
                    .and_then(|o| n.checked_add(o))
            })
            .and_then(|n| n.checked_add(2 * (256 << 10)))
            .and_then(|n| {
                retained_strings
                    .checked_mul(2)
                    .and_then(|s| n.checked_add(s))
            })
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

fn checked_phase_operations(layers: u64, per_layer: u64, fixed: u64) -> Result<usize, ModelError> {
    scalar_phase_operations(layers, per_layer, fixed).map_err(|_| rejected())
}
fn scalar_phase_operations(
    layers: u64,
    per_layer: u64,
    fixed: u64,
) -> Result<usize, SourceScalarSummaryError> {
    usize::try_from(
        layers
            .checked_mul(per_layer)
            .and_then(|n| n.checked_add(fixed))
            .ok_or(SourceScalarSummaryError::Overflow)?,
    )
    .map_err(|_| SourceScalarSummaryError::Overflow)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceScalarSummaryError {
    Unsupported,
    Invalid,
    Overflow,
}

/// Bounded logical scalar-query visits, not CPU fuel/opaque std internals.
/// Prepay before querying; this performs no source/model/native factory.
pub const SOURCE_SCALAR_SUMMARY_WORK: u64 = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceScalarSummary {
    pub layers: u64,
    pub hidden: u64,
    pub intermediate: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub query_width: u64,
    pub kv_width: u64,
    pub vocabulary: u64,
    pub shape_elements: [u64; 7],
    pub operations_per_phase: usize,
}
impl SourceScalarSummary {
    pub fn feature_width(&self) -> u64 {
        self.hidden
            .max(self.intermediate)
            .max(self.query_width)
            .max(self.kv_width)
    }
    pub fn semantic_elements(&self) -> u64 {
        self.shape_elements.iter().copied().max().unwrap_or(0)
    }
}

/// Exact scalar geometry using the registered preflight's head-width/default
/// contract; no family dispatch here and no parsing, allocation or construction.
pub fn source_scalar_summary(
    document: &Value,
    workload: DecoderWorkload,
) -> Result<SourceScalarSummary, SourceScalarSummaryError> {
    let mut summary = super::source_mappings::dense_scalar_summary(document, workload)?;
    let (layer_key, per_layer, _, fixed) =
        super::source_mappings::metadata_expansion_entry(document)
            .ok_or(SourceScalarSummaryError::Unsupported)?;
    summary.operations_per_phase =
        scalar_phase_operations(scalar_field(document, layer_key)?, per_layer, fixed)?;
    Ok(summary)
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

fn nonzero(value: u64) -> Result<u64, SourceScalarSummaryError> {
    if value == 0 {
        Err(SourceScalarSummaryError::Invalid)
    } else {
        Ok(value)
    }
}

fn field(document: &Value, key: &str) -> Result<u64, ModelError> {
    scalar_field(document, key).map_err(|_| rejected())
}
fn scalar_field(document: &Value, key: &str) -> Result<u64, SourceScalarSummaryError> {
    document
        .get(key)
        .and_then(Value::as_u64)
        .ok_or(SourceScalarSummaryError::Invalid)
}

fn optional_field(
    document: &Value,
    key: &str,
    fallback: u64,
) -> Result<u64, SourceScalarSummaryError> {
    match document.get(key) {
        None | Some(Value::Null) => Ok(fallback),
        Some(value) => value.as_u64().ok_or(SourceScalarSummaryError::Invalid),
    }
}

fn checked_product(values: &[u64]) -> Result<u64, SourceScalarSummaryError> {
    let value = values
        .iter()
        .try_fold(1u64, |total, value| total.checked_mul(*value))
        .ok_or(SourceScalarSummaryError::Overflow)?;
    nonzero(value)
}

/// Dimension layout shared by the registered ordinary dense source readers.
/// Their existing validators still check architectural and numeric semantics.
pub(super) fn preflight_dense_source(
    document: &Value,
    workload: DecoderWorkload,
    budget: DecoderExpansionBudget,
    head_width: HeadWidth,
) -> Result<(), ModelError> {
    // Legacy/nonregistered adapters retain the original conservative envelope.
    preflight_dense_readout_source(
        document,
        workload,
        budget,
        head_width,
        ReadoutContract::FullPrefix,
    )
}

/// Registered mapping evidence, not an execution mode or caller permission.
#[derive(Clone, Copy)]
pub(super) enum ReadoutContract {
    FullPrefix,
    LastToken,
}

pub(super) fn preflight_dense_readout_source(
    document: &Value,
    workload: DecoderWorkload,
    budget: DecoderExpansionBudget,
    head_width: HeadWidth,
    readout: ReadoutContract,
) -> Result<(), ModelError> {
    let summary =
        dense_scalar_summary(document, workload, head_width, readout).map_err(|_| rejected())?;
    positive(summary.layers, budget.max_layers)?;
    positive(summary.hidden, budget.max_feature_width)?;
    positive(summary.intermediate, budget.max_feature_width)?;
    positive(summary.attention_heads, budget.max_attention_heads)?;
    positive(summary.kv_heads, budget.max_attention_heads)?;
    positive(summary.head_dim, budget.max_feature_width)?;
    positive(summary.vocabulary, budget.max_vocabulary)?;
    positive(summary.query_width, budget.max_feature_width)?;
    positive(summary.kv_width, budget.max_feature_width)?;
    for elements in summary.shape_elements {
        positive(elements, budget.max_tensor_elements)?;
    }
    Ok(())
}

pub(super) fn dense_scalar_summary(
    document: &Value,
    workload: DecoderWorkload,
    head_width: HeadWidth,
    readout: ReadoutContract,
) -> Result<SourceScalarSummary, SourceScalarSummaryError> {
    let layers = nonzero(scalar_field(document, "num_hidden_layers")?)?;
    let hidden = nonzero(scalar_field(document, "hidden_size")?)?;
    let intermediate = nonzero(scalar_field(document, "intermediate_size")?)?;
    let heads = nonzero(scalar_field(document, "num_attention_heads")?)?;
    let kv_heads = nonzero(optional_field(document, "num_key_value_heads", heads)?)?;
    let head_dim = nonzero(match head_width {
        HeadWidth::HiddenPerHead => hidden / heads,
        HeadWidth::SourceField => scalar_field(document, "head_dim")?,
    })?;
    let vocabulary = nonzero(scalar_field(document, "vocab_size")?)?;
    let query_width = checked_product(&[heads, head_dim])?;
    let kv_width = checked_product(&[kv_heads, head_dim])?;
    if workload.batch == 0 || workload.max_input_tokens == 0 || workload.max_new_tokens == 0 {
        return Err(SourceScalarSummaryError::Invalid);
    }
    let capacity = workload
        .max_input_tokens
        .checked_add(workload.max_new_tokens)
        .and_then(|total| total.checked_sub(1))
        .ok_or(SourceScalarSummaryError::Overflow)?;
    let batch = workload.batch;
    let rows = workload.max_input_tokens;
    let readout_rows = match readout {
        ReadoutContract::FullPrefix => rows,
        ReadoutContract::LastToken => 1,
    };
    // Preflight is also used before an owned constructor has reserved its
    // payload. Shape inspection must not allocate temporary Vecs at that seam.
    let shapes = [
        &[batch, rows, hidden][..],
        &[batch, rows, intermediate][..],
        &[batch, readout_rows, vocabulary][..],
        &[batch, rows, heads, head_dim][..],
        &[batch, heads, rows, rows][..],
        &[batch, heads, capacity][..],
        &[batch, kv_heads, capacity, head_dim][..],
    ];
    let mut shape_elements = [0; 7];
    for (result, shape) in shape_elements.iter_mut().zip(shapes) {
        *result = checked_product(shape)?;
    }
    Ok(SourceScalarSummary {
        layers,
        hidden,
        intermediate,
        attention_heads: heads,
        kv_heads,
        head_dim,
        query_width,
        kv_width,
        vocabulary,
        shape_elements,
        operations_per_phase: 0,
    })
}

#[cfg(test)]
#[path = "source_readout_shape_tests.rs"]
mod source_readout_shape_tests;

#[cfg(test)]
mod factory_quote_tests {
    use super::*;
    use serde_json::json;

    fn source() -> Value {
        json!({"model_type":"qwen2", "num_hidden_layers":1, "hidden_size":4,
            "intermediate_size":8, "num_attention_heads":2, "num_key_value_heads":1,
            "vocab_size":7, "max_position_embeddings":8, "hidden_act":"silu",
            "rms_norm_eps":0.000001, "rope_theta":10000,
            "tie_word_embeddings":true, "use_sliding_window":false})
    }

    fn limits() -> ConfigLowererLimits {
        ConfigLowererLimits {
            max_operations_per_phase: 128,
            max_string_bytes: 1024,
            max_attribute_nodes: 32768,
            max_retained_strings: 1 << 20,
            max_factory_bytes: 64 << 20,
        }
    }

    fn expansion() -> DecoderExpansionBudget {
        DecoderExpansionBudget {
            max_layers: 8,
            max_feature_width: 128,
            max_attention_heads: 8,
            max_vocabulary: 128,
            max_tensor_elements: 32768,
        }
    }

    fn workload() -> DecoderWorkload {
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 4,
            max_new_tokens: 2,
        }
    }

    #[test]
    fn actual_factory_matches_independent_original_byte_lowerer() {
        let document = source();
        let original = crate::lower_model_json_with_budget(
            &serde_json::to_vec(&document).unwrap(),
            workload(),
            expansion(),
        )
        .unwrap();
        let quote = ConfigLowererQuote::new(&document, workload(), expansion(), limits()).unwrap();
        let admitted = quote.construct().unwrap();
        assert_eq!(admitted, original);
        assert_eq!(admitted.digest(), original.digest());
    }

    #[test]
    fn exact_layout_cap_and_one_short_before_factory() {
        let document = source();
        let quote = ConfigLowererQuote::new(&document, workload(), expansion(), limits()).unwrap();
        let bytes = quote.factory_bytes();
        let mut exact = limits();
        exact.max_factory_bytes = bytes;
        let exact_quote =
            ConfigLowererQuote::new(&document, workload(), expansion(), exact).unwrap();
        assert_eq!(exact_quote.factory_bytes(), bytes);
        let mut short = exact;
        short.max_factory_bytes -= 1;
        assert!(ConfigLowererQuote::new(&document, workload(), expansion(), short).is_err());
        let plan = exact_quote.construct().unwrap();
        assert!(plan.checked_owned_backing_bytes(bytes).unwrap() <= bytes);
    }

    #[test]
    fn genuine_dimensions_reject_before_semantic_factory() {
        let mut document = source();
        document["hidden_size"] = json!(1024);
        // The ordinary semantic factory has no caller width allowance; refuse
        // the actual opt-in dimensional limit before invoking that factory.
        assert!(ConfigLowererQuote::new(&document, workload(), expansion(), limits()).is_err());
    }
}
