//! Complete public client-local FP32 forward schedule. No protected authority.
use pllm_models::{ForwardDType, ForwardDimension, ForwardPlan, ForwardTensorSpec, ModelOperator};
use pllm_types::{canonical_bytes, canonical_digest, pipeline_digest_bytes, Digest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const FORWARD_RUNTIME_SCHEDULE_SCHEMA_VERSION: &str = "pllm.forward_runtime_schedule.v1";
const NUMERIC_DOMAIN: &str = "pllm.numeric.source_float32.v1";

/// Shared lexical input envelope; checked before either JSON parser allocates.
pub fn forward_input_preflight(bytes: &[u8]) -> Result<(), String> {
    crate::plan_input::public_json_preflight(bytes, crate::plan_input::Limits {
        max_source_bytes: 1 << 20, max_nodes: 65536, max_depth: 17,
        max_string_bytes: 1 << 20, max_number_bytes: 128,
        max_number_work: 1 << 28, max_work: 1 << 30,
        max_payload_bytes: 64 << 20,
    }).map(|_| ()).map_err(|error| error.to_string())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardRuntimeStep {
    pub operation_id: String,
    pub executor: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardRuntimeSchedule {
    pub schema_version: String,
    pub plan_digest: Digest,
    pub composition_digest: Digest,
    pub numeric_digest: Digest,
    pub max_sequence: u64,
    pub complete: bool,
    pub protected_execution: bool,
    pub steps: Vec<ForwardRuntimeStep>,
}

impl ForwardRuntimeSchedule {
    pub fn digest(&self) -> Digest {
        canonical_digest(FORWARD_RUNTIME_SCHEDULE_SCHEMA_VERSION, self)
    }
    /// Reconstruct, never accept a caller-chosen executor, operation order or digest.
    pub fn validate(&self, plan: &ForwardPlan, composition: &[u8]) -> Result<(), String> {
        self.validate_with_limits(plan, composition, pllm_models::default_forward_limits().map_err(|e| e.to_string())?)
    }
    pub fn validate_with_limits(&self, plan: &ForwardPlan, composition: &[u8], limits: pllm_models::ForwardLimits) -> Result<(), String> {
        if self != &lower_forward_runtime_schedule_with_limits(plan, composition, limits)? {
            return Err("forward schedule differs from reconstructed source schedule".into());
        }
        Ok(())
    }
}

pub fn lower_forward_runtime_schedule(
    plan: &ForwardPlan, composition: &[u8],
) -> Result<ForwardRuntimeSchedule, String> {
    lower_forward_runtime_schedule_with_limits(plan, composition, pllm_models::default_forward_limits().map_err(|e| e.to_string())?)
}

pub fn lower_forward_runtime_schedule_with_limits(
    plan: &ForwardPlan, composition: &[u8], limits: pllm_models::ForwardLimits,
) -> Result<ForwardRuntimeSchedule, String> {
    let pipeline = forward_composition(composition, limits)?;
    lower_forward_runtime_schedule_validated(plan, composition, &pipeline, limits)
}

/// Closed, bounded composition inspection. This does not grant source authority.
pub fn validate_forward_composition(composition: &[u8], limits: pllm_models::ForwardLimits) -> Result<(), String> {
    forward_composition(composition, limits).map(|_| ())
}

/// Disjoint logical frontend owners for bounded composition parsing/validation.
/// At most one token/member per input byte; actual typed/JSON tree layouts bound
/// headers and sixfold escaping bounds text. Four representations cover parser
/// Value, typed pipeline, canonical Value and output/sort/escape scratch. Source
/// parser workspace is reused sequentially by the caller, not included here.
pub fn forward_composition_memory_bytes(bytes: usize) -> Result<usize, String> {
    use std::mem::size_of;
    let node = pllm_models::owned_backing::checked_tree_node_bytes::<(String, serde_json::Value)>()
        .map_err(|e| e.to_string())?
        .max(pllm_models::owned_backing::checked_tree_node_bytes::<(String, crate::ExperimentComponent)>()
            .map_err(|e| e.to_string())?);
    let record = [node, size_of::<serde_json::Value>(), size_of::<String>(), size_of::<Vec<serde_json::Value>>(), 6]
        .iter().try_fold(0usize, |n, v| n.checked_add(*v).ok_or("forward composition record overflow"))?;
    bytes.checked_add(1).and_then(|n| n.checked_mul(record)).and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(size_of::<crate::ExperimentPipeline>()))
        .and_then(|n| n.checked_add("huggingface".len()))
        .ok_or_else(|| "forward composition owner overflow".into())
}

fn forward_composition(composition: &[u8], limits: pllm_models::ForwardLimits) -> Result<crate::ExperimentPipeline, String> {
    forward_input_preflight(composition)?;
    // Explicit parser phase bound, including corrected escaped decoding scratch.
    // The parser itself enforces this factory budget before Value allocation;
    // it is not a claim that the model Value-lifecycle peak includes raw parsing.
    let mut parsing = limits;
    parsing.metadata.max_factory_bytes = parsing.metadata.max_factory_bytes
        .min(forward_composition_memory_bytes(composition.len())?);
    let document = pllm_models::parse_bounded_source_json(composition, parsing, &[]).map_err(|e| e.to_string())?;
    let pipeline: crate::ExperimentPipeline = serde_json::from_value(document)
        .map_err(|_| "invalid forward composition")?;
    if canonical_bytes(&pipeline) != composition {
        return Err("forward composition must be canonical compact sorted JSON".into());
    }
    crate::validate_experiment_model(&pipeline.model)?;
    if !matches!(pipeline.model.kind.as_str(), "huggingface" | "safetensors") || pipeline.model.endpoint.is_some() {
        return Err("forward requires a client-owned source checkpoint, not a remote model backend".into());
    }
    let expected = [
        ("kernels", "pllm/cpu"),
        ("linear", "pllm/cleartext-linear"),
        ("numeric", "pllm/source-float32/v1"),
        ("topology", "pllm/client-only/v1"),
    ];
    if pipeline.components.len() != expected.len() || expected.iter().any(|(slot, id)| {
        pipeline.components.get(*slot).is_none_or(|c| c.component != *id || if *slot == "kernels" {
            c.params.len() != 1 || !matches!(c.params.get("threads").and_then(serde_json::Value::as_u64), Some(1..))
        } else { !c.params.is_empty() })
    }) {
        return Err("forward requires only bounded CPU threads, cleartext-linear, SourceFloat32 and client-only; no extra parameters".into());
    }
    Ok(pipeline)
}

fn lower_forward_runtime_schedule_validated(
    plan: &ForwardPlan, composition: &[u8], pipeline: &crate::ExperimentPipeline, limits: pllm_models::ForwardLimits,
) -> Result<ForwardRuntimeSchedule, String> {
    plan.validate_with_limits(limits).map_err(|e| e.to_string())?;
    if plan.max_sequence == 0 || plan.numeric != NUMERIC_DOMAIN {
        return Err("unsupported forward numeric contract".into());
    }
    let mut values: BTreeMap<&str, &ForwardTensorSpec> = plan.inputs.iter()
        .map(|(name, tensor)| (name.as_str(), tensor)).collect();
    let mut steps = Vec::new();
    steps.try_reserve_exact(plan.operations.len()).map_err(|_| "forward schedule allocation failed")?;
    for operation in &plan.operations {
        match operation.operator {
            ModelOperator::AttentionMask if pllm_models::is_key_padding_attention_mask(&operation.inputs, &operation.attributes) => {},
            ModelOperator::GeluErf if operation.inputs.len() == 1 && pllm_models::is_exact_fp32_gelu(&operation.attributes) => {},
            ModelOperator::Tanh if operation.inputs.len() == 1 && pllm_models::is_fp32_tanh(&operation.attributes) => {},
            ModelOperator::AttentionMask | ModelOperator::GeluErf | ModelOperator::Tanh => {
                return Err("unsupported forward numeric operator attributes".into());
            }
            ModelOperator::TokenLookup | ModelOperator::Linear | ModelOperator::LayerNorm
            | ModelOperator::ResidualAdd | ModelOperator::Reshape | ModelOperator::Permute
            | ModelOperator::AttentionScores | ModelOperator::AttentionScale
            | ModelOperator::Softmax | ModelOperator::AttentionValues | ModelOperator::Slice => {},
            _ => return Err("unsupported forward operator".into()),
        }
        if operation.inputs.iter().any(|name| !values.contains_key(name.as_str()))
            || values.contains_key(operation.id.as_str()) {
            return Err("forward graph is not topologically ordered with unique values".into());
        }
        tensor_bytes(&operation.output, plan.max_sequence)?;
        values.insert(&operation.id, &operation.output);
        steps.push(ForwardRuntimeStep {
            operation_id: operation.id.clone(), executor: "client_local_source_f32_v1".into(),
        });
    }
    for output in plan.outputs.values() {
        if values.get(output.value.as_str()).copied() != Some(&output.tensor) {
            return Err("forward named output differs from its typed source value".into());
        }
    }
    Ok(ForwardRuntimeSchedule {
        schema_version: FORWARD_RUNTIME_SCHEDULE_SCHEMA_VERSION.into(),
        plan_digest: plan.digest(), composition_digest: pipeline_digest_bytes(composition),
        numeric_digest: canonical_digest(NUMERIC_DOMAIN, &pipeline.components["numeric"]),
        max_sequence: plan.max_sequence, complete: true, protected_execution: false, steps,
    })
}

fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b).ok_or_else(|| "forward memory addition overflow".into())
}
fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b).ok_or_else(|| "forward memory multiplication overflow".into())
}
fn tensor_bytes(tensor: &ForwardTensorSpec, sequence: u64) -> Result<u64, String> {
    let mut count = 1;
    if tensor.shape.is_empty() { return Err("forward tensors must have positive rank".into()); }
    for dimension in &tensor.shape {
        let n = match dimension { ForwardDimension::Fixed(n) => *n, ForwardDimension::Sequence => sequence };
        if n == 0 { return Err("forward dimensions must be positive".into()); }
        count = mul(count, n)?;
    }
    mul(count, match tensor.dtype { ForwardDType::Int64 => 8, ForwardDType::Bool => 1, ForwardDType::Float32 => 4 })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ForwardMemoryQuote {
    pub sequence_length: u64,
    pub input_bytes: u64,
    pub retained_tensor_bytes: u64,
    pub output_bytes: u64,
    pub workspace_bytes: u64,
    pub metadata_bytes: u64,
    pub required_bytes: u64,
}

/// Descriptive scalar costs, not source provenance or executable admission
/// authority. Producers must account for actual typed storage and live roots.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ForwardMemoryInput {
    pub weight_bytes: u64,
    pub input_bytes: u64,
    pub largest_input_bytes: u64,
    pub retained_tensor_bytes: u64,
    pub largest_operation_bytes: u64,
    pub output_bytes: u64,
}

/// Current source-FP32 model summaries describe numerical bodies in f32 elements;
/// input widths are already typed bytes. No source-family geometry is parsed here.
pub fn forward_memory_input_from_summary(
    summary: &pllm_models::ForwardNumericalSummary,
) -> Result<ForwardMemoryInput, String> {
    Ok(ForwardMemoryInput {
        weight_bytes: mul(summary.parameter_elements, 4)?,
        input_bytes: summary.input_bytes,
        largest_input_bytes: summary.largest_input_bytes,
        retained_tensor_bytes: mul(summary.operation_output_elements, 4)?,
        largest_operation_bytes: mul(summary.largest_operation_elements, 4)?,
        output_bytes: mul(summary.public_output_elements, 4)?,
    })
}

/// Extract costs from the expanded typed graph. This is an inspection adapter;
/// it neither validates source provenance nor grants execution authority.
pub fn forward_memory_input_from_plan(
    plan: &ForwardPlan, sequence: u64, weight_bytes: u64,
) -> Result<ForwardMemoryInput, String> {
    if sequence == 0 || sequence > plan.max_sequence { return Err("sequence_length outside forward bound".into()); }
    let mut input = ForwardMemoryInput { weight_bytes, ..ForwardMemoryInput::default() };
    for tensor in plan.inputs.values() {
        let n = tensor_bytes(tensor, sequence)?;
        input.input_bytes = add(input.input_bytes, n)?;
        input.largest_input_bytes = input.largest_input_bytes.max(n);
    }
    for operation in &plan.operations {
        let n = tensor_bytes(&operation.output, sequence)?;
        input.retained_tensor_bytes = add(input.retained_tensor_bytes, n)?;
        input.largest_operation_bytes = input.largest_operation_bytes.max(n);
    }
    for output in plan.outputs.values() {
        input.output_bytes = add(input.output_bytes, tensor_bytes(&output.tensor, sequence)?)?;
    }
    Ok(input)
}

/// Shared method-neutral backend policy. Logical numerical storage, conservative
/// kernel scratch and explicit metadata charge, NOT process RSS. No caller cap
/// influences this deterministic quote. Weight loading/snapshot overlap and
/// ordinary host admission remain separate.
pub fn quote_forward_memory_input(
    sequence: u64, input: &ForwardMemoryInput, metadata_bytes: u64,
) -> Result<ForwardMemoryQuote, String> {
    if sequence == 0 { return Err("sequence_length must be positive".into()); }
    let largest = input.largest_input_bytes.max(input.largest_operation_bytes);
    // Eight full-size FP32/F64/boolean scratch equivalents exceed all admitted
    // elementwise, centered variance, softmax and bounded activation phases.
    // Include a second weight-sized scratch for layout conversion overlap.
    // Additional input-sized copies cover normalization, immutable snapshots,
    // decode bridges and publication overlap even for narrow hidden geometry.
    // Eight bounded 1-MiB chunk equivalents price FFI serialization, finite
    // boolean/reduction buffers and ordinary unoptimized NumPy loop scratch.
    // This allowance is not allocator/RSS accounting or a zero-internal-allocation proof.
    let workspace_bytes = add(add(add(mul(largest, 8)?, mul(input.input_bytes, 4)?)?, input.weight_bytes)?, 8 << 20)?;
    let numerical_bytes = add(add(add(add(input.weight_bytes, input.input_bytes)?, input.retained_tensor_bytes)?, input.output_bytes)?, workspace_bytes)?;
    let required_bytes = add(numerical_bytes, metadata_bytes)?;
    usize::try_from(required_bytes).map_err(|_| "forward memory exceeds address space")?;
    Ok(ForwardMemoryQuote {
        sequence_length: sequence, input_bytes: input.input_bytes,
        retained_tensor_bytes: input.retained_tensor_bytes, output_bytes: input.output_bytes,
        workspace_bytes, metadata_bytes, required_bytes,
    })
}

/// Existing numerical-only graph quote. Retains every operation root, including
/// materialized aliases, and prices named immutable result publication separately.
/// Metadata remains zero until native admission supplies its explicit charge.
pub fn quote_forward_memory(plan: &ForwardPlan, sequence: u64, weight_bytes: u64) -> Result<ForwardMemoryQuote, String> {
    let input = forward_memory_input_from_plan(plan, sequence, weight_bytes)?;
    quote_forward_memory_input(sequence, &input, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn source() -> Vec<u8> {
        canonical_bytes(&json!({"model_type":"bert","architectures":["BertModel"],
            "vocab_size":17,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
            "intermediate_size":12,"max_position_embeddings":16,"type_vocab_size":2,
            "hidden_act":"gelu","layer_norm_eps":"1e-12","position_embedding_type":"absolute",
            "torch_dtype":"float32"}))
    }
    fn composition() -> Vec<u8> {
        canonical_bytes(&json!({"model":{"source":"fixture"},"components":{
            "kernels":{"component":"pllm/cpu","params":{"threads":1}},
            "linear":{"component":"pllm/cleartext-linear","params":{}},
            "numeric":{"component":"pllm/source-float32/v1","params":{}},
            "topology":{"component":"pllm/client-only/v1","params":{}}}}))
    }
    #[test]
    fn schedule_reconstruction_and_symbolic_axes() {
        let plan = pllm_models::lower_forward_model(&source(), 8).unwrap();
        let schedule = lower_forward_runtime_schedule(&plan, &composition()).unwrap();
        let limits = pllm_models::forward_limits_for_memory(1 << 30).unwrap();
        assert_eq!(schedule, lower_forward_runtime_schedule_with_limits(&plan, &composition(), limits).unwrap());
        schedule.validate_with_limits(&plan, &composition(), limits).unwrap();
        assert!(schedule.complete && !schedule.protected_execution);
        assert_eq!(schedule.steps.len(), plan.operations.len());
        assert!(plan.operations.iter().filter(|op| op.operator == ModelOperator::AttentionScores)
            .all(|op| op.output.shape[2..] == [ForwardDimension::Sequence, ForwardDimension::Sequence]));
        schedule.validate(&plan, &composition()).unwrap();
        let mut tampered = schedule.clone();
        tampered.steps[0].executor = "remote_stage".into();
        assert!(tampered.validate(&plan, &composition()).is_err());
        let mut tampered = schedule.clone();
        tampered.steps.swap(0, 1);
        assert!(tampered.validate(&plan, &composition()).is_err());
        let mut plan = plan.clone();
        plan.operations[0].attributes["weight"] = json!("forged");
        assert!(lower_forward_runtime_schedule(&plan, &composition()).is_err());
    }
    #[test]
    fn raw_composition_and_decoder_refusal() {
        let plan = pllm_models::lower_forward_model(&source(), 8).unwrap();
        assert!(crate::classify_decoder_composition(&composition()).is_err());
        for slot in ["state", "scheduler", "preparation", "verification", "quantization", "cache", "private"] {
            let mut pipeline: serde_json::Value = serde_json::from_slice(&composition()).unwrap();
            pipeline["components"][slot] = json!({"component":"extra","params":{}});
            assert!(lower_forward_runtime_schedule(&plan, &canonical_bytes(&pipeline)).is_err());
        }
        let mut pipeline: serde_json::Value = serde_json::from_slice(&composition()).unwrap();
        pipeline["components"]["numeric"]["params"] = json!({"bits":32});
        assert!(lower_forward_runtime_schedule(&plan, &canonical_bytes(&pipeline)).is_err());
    }
    #[test]
    fn variable_s_checked_quote_and_overflow() {
        let mut plan = pllm_models::lower_forward_model(&source(), 8).unwrap();
        let small = quote_forward_memory(&plan, 1, 100).unwrap();
        let large = quote_forward_memory(&plan, 8, 100).unwrap();
        assert_eq!(large.input_bytes, 200);
        assert!(large.required_bytes > small.required_bytes);
        assert!(quote_forward_memory(&plan, 0, 100).is_err());
        assert!(quote_forward_memory(&plan, 9, 100).is_err());
        assert!(quote_forward_memory(&plan, 8, u64::MAX).is_err());
        plan.operations[0].output.shape = vec![ForwardDimension::Fixed(u64::MAX), ForwardDimension::Sequence];
        assert!(quote_forward_memory(&plan, 8, 0).is_err());
    }

    #[test]
    fn neutral_memory_policy_exact_formula_metadata_and_cap_independence() {
        let input = ForwardMemoryInput {
            weight_bytes: 100, input_bytes: 25, largest_input_bytes: 8,
            retained_tensor_bytes: 200, largest_operation_bytes: 40, output_bytes: 32,
        };
        let quote = quote_forward_memory_input(3, &input, 123).unwrap();
        assert_eq!(quote, ForwardMemoryQuote {
            sequence_length: 3, input_bytes: 25, retained_tensor_bytes: 200,
            output_bytes: 32, workspace_bytes: 8 * 40 + 4 * 25 + 100 + (8 << 20),
            metadata_bytes: 123,
            required_bytes: 100 + 25 + 200 + 32 + 8 * 40 + 4 * 25 + 100 + (8 << 20) + 123,
        });
        let document = serde_json::to_value(&quote).unwrap();
        assert_eq!(document["metadata_bytes"], json!(123));
        // Caps are admission decisions outside the cost function. Neither a
        // rejecting cap nor an accepting cap may alter its returned costs.
        for cap in [quote.required_bytes - 1, quote.required_bytes, quote.required_bytes + 1] {
            let repeated = quote_forward_memory_input(3, &input, 123).unwrap();
            assert_eq!(repeated, quote);
            assert_eq!(repeated.required_bytes <= cap, cap >= quote.required_bytes);
        }
        let uncharged = quote_forward_memory_input(3, &input, 0).unwrap();
        assert_eq!(quote.required_bytes - uncharged.required_bytes, 123);
        assert_eq!(quote.workspace_bytes, uncharged.workspace_bytes);
        assert!(quote_forward_memory_input(0, &input, 123).is_err());
        let input_largest = ForwardMemoryInput {
            largest_input_bytes: 80, ..input
        };
        assert_eq!(quote_forward_memory_input(3, &input_largest, 0).unwrap().workspace_bytes,
            uncharged.workspace_bytes + 8 * (80 - 40));
    }

    #[test]
    fn neutral_memory_policy_checked_overflow_at_multiple_terms() {
        let zero = ForwardMemoryInput::default();
        let cases = [
            ForwardMemoryInput { largest_input_bytes: u64::MAX / 8 + 1, ..zero },
            ForwardMemoryInput { largest_operation_bytes: u64::MAX / 8 + 1, ..zero },
            ForwardMemoryInput { input_bytes: u64::MAX / 4 + 1, ..zero },
            ForwardMemoryInput { weight_bytes: u64::MAX, largest_operation_bytes: 1, ..zero },
            ForwardMemoryInput { weight_bytes: u64::MAX - (8 << 20), ..zero },
            ForwardMemoryInput { retained_tensor_bytes: u64::MAX, output_bytes: 1, ..zero },
            ForwardMemoryInput { output_bytes: u64::MAX, ..zero },
        ];
        for input in cases {
            assert!(quote_forward_memory_input(1, &input, 0).is_err());
        }
        assert!(quote_forward_memory_input(1, &zero, u64::MAX).is_err());
        let valid = quote_forward_memory_input(1, &zero, 7).unwrap();
        assert_eq!(valid.required_bytes, (8 << 20) + 7);
        assert!(usize::try_from(valid.required_bytes).is_ok());
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn neutral_memory_policy_rejects_address_space_without_u64_overflow() {
        let input = ForwardMemoryInput {
            retained_tensor_bytes: u64::from(u32::MAX) + 1,
            ..ForwardMemoryInput::default()
        };
        assert!(quote_forward_memory_input(1, &input, 0).is_err());
    }

    #[test]
    fn expanded_graph_adapter_preserves_numerical_quote_and_quadratic_attention() {
        let plan = pllm_models::lower_forward_model(&source(), 8).unwrap();
        for sequence in [1, 2, 3, 8] {
            let input = forward_memory_input_from_plan(&plan, sequence, 100).unwrap();
            let quote = quote_forward_memory(&plan, sequence, 100).unwrap();
            assert_eq!(quote, quote_forward_memory_input(sequence, &input, 0).unwrap());
            assert_eq!(quote.metadata_bytes, 0);
            assert_eq!(input.input_bytes, 25 * sequence);
            assert_eq!(input.largest_input_bytes, 8 * sequence);
            // Independent spelling of the old numerical backend formula.
            let largest = input.largest_input_bytes.max(input.largest_operation_bytes);
            let old_workspace = 8 * largest + 4 * input.input_bytes + input.weight_bytes + (8 << 20);
            let old_required = input.weight_bytes + input.input_bytes + input.retained_tensor_bytes
                + input.output_bytes + old_workspace;
            assert_eq!(quote.workspace_bytes, old_workspace);
            assert_eq!(quote.required_bytes, old_required);
            assert!(quote_forward_memory_input(sequence, &input, 55).unwrap().required_bytes == old_required + 55);
        }
        let score = plan.operations.iter().find(|op| op.operator == ModelOperator::AttentionScores).unwrap();
        assert_eq!(tensor_bytes(&score.output, 2).unwrap(), 4 * tensor_bytes(&score.output, 1).unwrap());
        assert_eq!(tensor_bytes(&score.output, 3).unwrap(), 9 * tensor_bytes(&score.output, 1).unwrap());
        let a = forward_memory_input_from_plan(&plan, 1, 0).unwrap();
        let b = forward_memory_input_from_plan(&plan, 2, 0).unwrap();
        let c = forward_memory_input_from_plan(&plan, 3, 0).unwrap();
        assert!(c.retained_tensor_bytes + a.retained_tensor_bytes > 2 * b.retained_tensor_bytes);
    }

    #[test]
    fn expanded_graph_adapter_accounts_for_actual_non_f32_tensor_types() {
        // Costs only: mutating descriptive shapes/dtypes does not mint a source
        // contract, schedule or opaque execution owner.
        let mut plan = pllm_models::lower_forward_model(&source(), 8).unwrap();
        let before = forward_memory_input_from_plan(&plan, 2, 0).unwrap();
        let op_elements = tensor_bytes(&plan.operations[0].output, 2).unwrap() / 4;
        plan.operations[0].output.dtype = ForwardDType::Bool;
        let output = plan.outputs.get_mut("pooler_output").unwrap();
        let output_elements = tensor_bytes(&output.tensor, 2).unwrap() / 4;
        output.tensor.dtype = ForwardDType::Int64;
        let after = forward_memory_input_from_plan(&plan, 2, 0).unwrap();
        assert_eq!(after.retained_tensor_bytes, before.retained_tensor_bytes - 3 * op_elements);
        assert_eq!(after.output_bytes, before.output_bytes + 4 * output_elements);
        assert_eq!(after.input_bytes, before.input_bytes);
    }

    #[test]
    fn model_summary_matches_expanded_typed_graph_across_source_shapes() {
        for classifier in [false, true] {
            for (layers, intermediate) in [(1, 12), (2, 13), (3, 5)] {
                let mut config: serde_json::Value = serde_json::from_slice(&source()).unwrap();
                config["num_hidden_layers"] = json!(layers);
                config["intermediate_size"] = json!(intermediate);
                if classifier {
                    config["architectures"] = json!(["BertForSequenceClassification"]);
                    config["num_labels"] = json!(3);
                }
                for sequence in [1, 8] {
                    let preflight = pllm_models::preflight_forward_value(&config, sequence,
                        pllm_models::default_forward_limits().unwrap()).unwrap();
                    let resource = *preflight.quote();
                    let expected = forward_memory_input_from_summary(&resource.numerical).unwrap();
                    let plan = preflight.construct().unwrap();
                    let requirements = pllm_models::forward_checkpoint_requirements(&plan).unwrap();
                    let weight_bytes: u64 = requirements.iter().map(|tensor| {
                        assert_eq!(tensor.dtype, ForwardDType::Float32);
                        tensor.shape.iter().product::<u64>() * 4
                    }).sum();
                    assert_eq!(expected, forward_memory_input_from_plan(&plan, sequence, weight_bytes).unwrap());
                    assert_eq!(resource.operation_count, plan.operations.len());
                    assert_eq!(resource.operation_id_bytes, plan.operations.iter().map(|op| op.id.len()).sum::<usize>());
                }
            }
        }
    }

    #[test]
    fn model_summary_scalar_adapter_checks_element_to_byte_overflow() {
        let mut summary = pllm_models::ForwardNumericalSummary {
            parameter_elements: 1, operation_output_elements: 2, largest_operation_elements: 2,
            input_bytes: 25, largest_input_bytes: 8, public_output_elements: 1,
        };
        assert_eq!(forward_memory_input_from_summary(&summary).unwrap().weight_bytes, 4);
        summary.parameter_elements = u64::MAX / 4 + 1;
        assert!(forward_memory_input_from_summary(&summary).is_err());
        summary.parameter_elements = 0;
        summary.operation_output_elements = u64::MAX / 4 + 1;
        assert!(forward_memory_input_from_summary(&summary).is_err());
        summary.operation_output_elements = 0;
        summary.largest_operation_elements = u64::MAX / 4 + 1;
        assert!(forward_memory_input_from_summary(&summary).is_err());
        summary.largest_operation_elements = 0;
        summary.public_output_elements = u64::MAX / 4 + 1;
        assert!(forward_memory_input_from_summary(&summary).is_err());
    }

    #[test]
    fn composition_owner_bound_uses_checked_input_counts() {
        let small = forward_composition_memory_bytes(1).unwrap();
        assert!(forward_composition_memory_bytes(2).unwrap() > small);
        assert!(forward_composition_memory_bytes(usize::MAX).is_err());
    }

    #[test]
    fn restrictive_supplied_operation_limits_do_not_fall_back_to_default_validation() {
        let plan = pllm_models::lower_forward_model(&source(), 8).unwrap();
        let schedule = lower_forward_runtime_schedule(&plan, &composition()).unwrap();
        let mut limits = pllm_models::default_forward_limits().unwrap();
        limits.metadata.max_operations_per_phase = plan.operations.len() - 1;
        // Isolate model reconstruction limit, not an unrelated parser refusal.
        validate_forward_composition(&composition(), limits).unwrap();
        assert!(lower_forward_runtime_schedule_with_limits(&plan, &composition(), limits).is_err());
        assert!(schedule.validate_with_limits(&plan, &composition(), limits).is_err());
        schedule.validate(&plan, &composition()).unwrap();
    }

    #[test]
    fn compiler_composition_helper_refuses_duplicate_unknown_nonfinite_and_tight_parser_inputs() {
        let limits = pllm_models::default_forward_limits().unwrap();
        let raw = String::from_utf8(composition()).unwrap();
        let duplicate = raw.replace("\"threads\":1", "\"threads\":1,\"threads\":1");
        assert_ne!(duplicate, raw);
        assert!(validate_forward_composition(duplicate.as_bytes(), limits).is_err());
        for path in [vec![], vec!["model"], vec!["components", "numeric"]] {
            let mut document: serde_json::Value = serde_json::from_str(&raw).unwrap();
            let mut target = &mut document;
            for key in path { target = &mut target[key]; }
            target["unknown"] = json!(true);
            assert!(validate_forward_composition(&canonical_bytes(&document), limits).is_err());
        }
        for number in ["NaN", "Infinity", "-Infinity", "1e9999"] {
            let malformed = raw.replace("\"threads\":1", &format!("\"threads\":{number}"));
            assert!(validate_forward_composition(malformed.as_bytes(), limits).is_err());
        }
        let mut tight = limits;
        tight.metadata.max_factory_bytes = 1;
        assert!(validate_forward_composition(raw.as_bytes(), tight).is_err());
    }
}
