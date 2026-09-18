//! Deterministic lowering from locked context into canonical plans plus a private region program.

use pllm_models::{
    DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator, StateKind,
};
use pllm_types::{
    assurance_result_digest, canonical_bytes, canonical_digest, configuration_digest_bytes,
    digest_bytes, execution_plan_digest, logical_plan_digest, plan_lock_bytes,
    privacy_contract_digest, valid_identity, AssuranceResult, Digest, EvidenceReference,
    ExecutionPlan, LockedContext, LogicalPlan, NamedDigest, PlanLock, PrivacyContract,
    ResolvedComponent, RolePlanReference, VersionedArtifact, ASSURANCE_RESULT_SCHEMA_VERSION,
    EXECUTION_PLAN_SCHEMA_VERSION, LOCKED_CONTEXT_SCHEMA_VERSION, LOGICAL_PLAN_SCHEMA_VERSION,
    PLAN_LOCK_SCHEMA_VERSION, PRIVACY_CONTRACT_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};

pub use pllm_garble::{prepare_silu_quadratic_q7, SiluQuadraticQ7Material};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Mutex, OnceLock};

mod dense_qwen_mlp;
mod dense_qwen_mlp_protected;
mod gated_tensor;
mod provenance_primitives;
mod rms_norm_protected;
mod rms_norm_stream_protected;
pub use dense_qwen_mlp::{
    compile_dense_qwen_mlp_block, execute_dense_qwen_mlp_block, CompiledDenseQwenMlpBlock,
    DenseQwenMlpComposite, DenseQwenMlpElementType, DenseQwenMlpLayout, DenseQwenMlpRangePolicy,
    DenseQwenMlpResidualOutputRangePolicy, DenseQwenMlpResidualRefinement,
    DenseQwenMlpResidualStorage, DenseQwenMlpResourcePolicy, DenseQwenMlpWeightArtifact,
    DenseQwenMlpWeightBytes, DenseQwenMlpWeightManifest, DenseQwenMlpWeights,
    DENSE_QWEN_MLP_BLOCK_SCHEMA_VERSION, DENSE_QWEN_MLP_CLEAR_PROFILE,
    DENSE_QWEN_MLP_HARD_MAX_ACTIVATION_ELEMENTS, DENSE_QWEN_MLP_HARD_MAX_TOTAL_WEIGHT_BYTES,
    DENSE_QWEN_MLP_WEIGHT_MANIFEST_SCHEMA_VERSION,
};
pub use dense_qwen_mlp_protected::{
    execute_bound_dense_qwen_mlp_protected_nonlinear,
    prepare_bound_dense_qwen_mlp_protected_nonlinear, BoundDenseQwenMlpProtectedMaterial,
    DenseQwenMlpProtectedNonlinearBinding, DenseQwenMlpProtectedNonlinearExecution,
    DenseQwenMlpProtectedResourcePolicy, ExperimentalDenseQwenMlpProtectedNonlinearApproval,
    DENSE_QWEN_MLP_PROTECTED_HARD_MAX_ROWS, DENSE_QWEN_MLP_PROTECTED_HARD_MAX_TOTAL_BODY_BYTES,
};
pub use gated_tensor::{
    prepare_bound_gated_multiply_q7_tensor_material, BoundGatedMultiplyQ7TensorMaterial,
    GatedMultiplyQ7TensorEvaluator, GatedMultiplyQ7TensorTicket, TensorResourcePolicy,
};
pub use provenance_primitives::{
    append_model_kv_cache_q10, execute_model_kv_cache_view_q10, execute_model_rope_q10,
    initialize_model_kv_cache_q10, lower_model_kv_cache_append_q10_regions,
    lower_model_kv_cache_view_q10_regions, lower_model_rope_q10_regions,
    provenance_primitives_artifact_digest, BoundedKvCacheViewQ10, KvCacheAppendMode,
    KvCacheQ10ExecutionInputs, ModelKvCacheAppendQ10Region, ModelKvCacheViewQ10Region,
    ModelRopeQ10Region, ProvenanceBoundKvCacheQ10, ProvenancePrimitiveResourcePolicy,
    KV_CACHE_APPEND_Q10_REGION_SCHEMA_VERSION, KV_CACHE_Q10_ACTIVE_POLICY,
    KV_CACHE_Q10_ZERO_PADDING_POLICY, KV_CACHE_VIEW_Q10_REGION_SCHEMA_VERSION,
    PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES, PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_BYTES,
    PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_DEPTH, PROVENANCE_PRIMITIVE_HARD_MAX_OPERATIONS,
    PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS, PROVENANCE_PRIMITIVE_HARD_MAX_STATES,
    PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES, ROPE_Q10_POSITION_LAYOUT, ROPE_Q10_RANGE_POLICY,
    ROPE_Q10_REGION_SCHEMA_VERSION,
};
pub use rms_norm_protected::{
    prepare_bound_rms_norm_q10_row, BoundRmsNormQ10ClientMaterial, BoundRmsNormQ10Decoder,
    BoundRmsNormQ10Evaluation, BoundRmsNormQ10Outputs, ExperimentalRmsNormQ10Policy,
};
pub use rms_norm_stream_protected::{
    prepare_bound_rms_norm_q10_stream_row, BoundRmsNormQ10StreamDecoder,
    BoundRmsNormQ10StreamEvaluator, BoundRmsNormQ10StreamInputs, BoundRmsNormQ10StreamMaterial,
    BoundRmsNormQ10StreamOutputs, ExperimentalRmsNormQ10StreamPolicy,
    RmsNormQ10StreamResourcePolicy, RmsNormQ10StreamTicket,
};

pub const REGION_PROGRAM_SCHEMA_VERSION: &str = "pllm.region_program.v1";
pub const COMPILE_REQUEST_SCHEMA_VERSION: &str = "pllm.compile_request.v1";
pub const BASELINE_EXPERIMENT_PROFILE: &str = "baseline.masked_linear_cpu";
pub const SILU_Q7_EXPERIMENT_PROFILE: &str = "research.single_evaluator";
pub const SILU_Q7_NUMERIC_GRAPH_ID: &str = pllm_core::activation::SILU_QUADRATIC_Q7_PROFILE;
pub const SILU_Q7_PROTECTED_GRAPH_ID: &str = "pllm.protected.arithmetic_garbling.silu_q7.v1";
pub const SILU_Q7_METHOD_ID: &str = "arithmetic-garbling-silu-q7";
pub const SILU_Q7_KERNEL_DESCRIPTOR_ID: &str = "pllm-garble-silu-quadratic-q7";
pub const SILU_Q7_COMPILER_ID: &str = "pllm-compiler";
pub const SILU_Q7_MAX_TENSOR_ELEMENTS: usize = 128;
pub const SILU_Q7_MAX_EVALUATOR_PAYLOAD_BYTES: usize = 16_384;
pub const SILU_Q7_MAX_LABEL_BYTES: usize = 1_024;
pub const Q14_TO_Q7_REGION_SCHEMA_VERSION: &str = "pllm.numeric.rescale_region.v2";
pub const RMS_NORM_F32_DIRECT_REGION_SCHEMA_VERSION: &str = "pllm.rms_norm_f32_direct_region.v1";
pub const RMS_NORM_Q10_DIRECT_REGION_SCHEMA_VERSION: &str = "pllm.rms_norm_q10_direct_region.v1";
pub const GATED_MULTIPLY_Q7_REGION_SCHEMA_VERSION: &str = "pllm.gated_multiply_q7_region.v2";
pub const GATED_MULTIPLY_Q7_NUMERIC_GRAPH_ID: &str =
    pllm_core::fixed_point::GATED_MULTIPLY_Q7_PROFILE;
pub const GATED_MULTIPLY_Q7_PROTECTED_GRAPH_ID: &str =
    "pllm.protected.arithmetic_garbling.gated_multiply_q7.v1";
pub const GATED_MULTIPLY_Q7_METHOD_ID: &str = "arithmetic-garbling-gated-multiply-q7";
pub const GATED_MULTIPLY_Q7_KERNEL_ID: &str = "pllm-garble-gated-multiply-q7";
pub const GATED_MULTIPLY_Q7_BINARY_TABLE_COMPONENT_ID: &str =
    pllm_garble::BINARY_TABLE_GATED_MULTIPLY_Q7_COMPONENT_ID;
pub const GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID: &str =
    pllm_garble::R03_CRT_GATED_MULTIPLY_Q7_COMPONENT_ID;
pub const GATED_MULTIPLY_Q7_SCALAR_SCHEDULE_COMPONENT_ID: &str = "pllm/scalar/v1";
pub const GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID: &str =
    "pllm/independent-lanes/v1";
pub const GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID: &str =
    "pllm/chunked-independent-lanes/v1";
pub const GATED_MULTIPLY_Q7_MAX_TENSOR_ELEMENTS: usize = 4;
pub const GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS: usize = 4_000_000;
pub const GATED_MULTIPLY_Q7_MAX_EVALUATOR_PAYLOAD_BYTES: usize = 10_000_000;
const SILU_Q7_ISSUANCE_CAPACITY: usize = 65_536;
const GATED_MULTIPLY_Q7_ISSUANCE_CAPACITY: usize = 1_024;
const SILU_Q7_GATE_SCHEMA_VERSION: &str = "pllm.silu_q7_gate.v2";
const SILU_Q7_ISSUANCE_DIGEST_DOMAIN: &str = "pllm.silu_q7_gate.issuance.v1";
const GATED_MULTIPLY_Q7_PROGRAM_SCHEMA_VERSION: &str = "pllm.gated_multiply_q7_bundle.v1";
const GATED_MULTIPLY_Q7_ISSUANCE_DIGEST_DOMAIN: &str = "pllm.gated_multiply_q7_gate.issuance.v1";
const GATED_MULTIPLY_Q7_LANE_CONTEXT_DOMAIN: &str = "pllm.gated_multiply_q7_lane.v1";
const GATED_MULTIPLY_Q7_BUNDLE_ID_DOMAIN: &str = "pllm.gated_multiply_q7_bundle_id.v1";

pub fn silu_q7_kernel_artifact_digest() -> Digest {
    canonical_digest(
        "pllm.artifact.rust-source-set.v1",
        &[
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-core/src/activation.rs"),
            ),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-core/src/fixed_point.rs"),
            ),
            silu_q7_method_artifact_digest(),
        ],
    )
}

pub fn silu_q7_compiler_artifact_digest() -> Digest {
    canonical_digest(
        "pllm.artifact.rust-source-set.v1",
        &[
            digest_bytes("pllm.artifact.rust-source.v1", include_bytes!("lib.rs")),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("gated_tensor.rs"),
            ),
        ],
    )
}

pub fn rms_norm_kernel_artifact_digest() -> Digest {
    canonical_digest(
        "pllm.artifact.rust-source-set.v1",
        &[
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-core/src/rms_norm.rs"),
            ),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-core/src/fixed_point.rs"),
            ),
        ],
    )
}

pub fn protected_rms_norm_q10_artifact_digest() -> Digest {
    #[derive(Serialize)]
    struct ArtifactFile<'a> {
        path: &'a str,
        digest: Digest,
    }

    let file = |path, bytes: &[u8]| ArtifactFile {
        path,
        digest: digest_bytes("pllm.artifact.file.v1", bytes),
    };
    canonical_digest(
        "pllm.artifact.rms_norm_q10.complete_source_set.v1",
        &[
            file(
                "crates/pllm-garble/src/boolean.rs",
                include_bytes!("../../pllm-garble/src/boolean.rs"),
            ),
            file(
                "crates/pllm-garble/src/lib.rs",
                include_bytes!("../../pllm-garble/src/lib.rs"),
            ),
            file(
                "crates/pllm-garble/src/boolean_stream.rs",
                include_bytes!("../../pllm-garble/src/boolean_stream.rs"),
            ),
            file(
                "crates/pllm-core/src/lib.rs",
                include_bytes!("../../pllm-core/src/lib.rs"),
            ),
            file(
                "crates/pllm-core/src/rms_norm.rs",
                include_bytes!("../../pllm-core/src/rms_norm.rs"),
            ),
            file(
                "crates/pllm-core/src/fixed_point.rs",
                include_bytes!("../../pllm-core/src/fixed_point.rs"),
            ),
            file(
                "crates/pllm-models/src/lib.rs",
                include_bytes!("../../pllm-models/src/lib.rs"),
            ),
            file(
                "crates/pllm-models/src/cache.rs",
                include_bytes!("../../pllm-models/src/cache.rs"),
            ),
            file(
                "crates/pllm-models/src/gemma4.rs",
                include_bytes!("../../pllm-models/src/gemma4.rs"),
            ),
            file(
                "crates/pllm-models/src/phi4.rs",
                include_bytes!("../../pllm-models/src/phi4.rs"),
            ),
            file(
                "crates/pllm-models/src/qwen35.rs",
                include_bytes!("../../pllm-models/src/qwen35.rs"),
            ),
            file(
                "crates/pllm-types/src/lib.rs",
                include_bytes!("../../pllm-types/src/lib.rs"),
            ),
            file("crates/pllm-compiler/src/lib.rs", include_bytes!("lib.rs")),
            file(
                "crates/pllm-compiler/src/rms_norm_protected.rs",
                include_bytes!("rms_norm_protected.rs"),
            ),
            file(
                "crates/pllm-compiler/src/rms_norm_stream_protected.rs",
                include_bytes!("rms_norm_stream_protected.rs"),
            ),
            file("Cargo.lock", include_bytes!("../../../Cargo.lock")),
            file("Cargo.toml", include_bytes!("../../../Cargo.toml")),
            file(
                "crates/pllm-assurance/Cargo.toml",
                include_bytes!("../../pllm-assurance/Cargo.toml"),
            ),
            file(
                "crates/pllm-bench/Cargo.toml",
                include_bytes!("../../pllm-bench/Cargo.toml"),
            ),
            file(
                "crates/pllm-compiler/Cargo.toml",
                include_bytes!("../Cargo.toml"),
            ),
            file(
                "crates/pllm-core/Cargo.toml",
                include_bytes!("../../pllm-core/Cargo.toml"),
            ),
            file(
                "crates/pllm-garble/Cargo.toml",
                include_bytes!("../../pllm-garble/Cargo.toml"),
            ),
            file(
                "crates/pllm-models/Cargo.toml",
                include_bytes!("../../pllm-models/Cargo.toml"),
            ),
            file(
                "crates/pllm-python/Cargo.toml",
                include_bytes!("../../pllm-python/Cargo.toml"),
            ),
            file(
                "crates/pllm-types/Cargo.toml",
                include_bytes!("../../pllm-types/Cargo.toml"),
            ),
            file(
                "rust-toolchain.toml",
                include_bytes!("../../../rust-toolchain.toml"),
            ),
        ],
    )
}

#[derive(Serialize)]
pub struct SiluQ7InstalledContract {
    pub profile: &'static str,
    pub compiler_id: &'static str,
    pub compiler_version: &'static str,
    pub compiler_artifact_digest: Digest,
    pub numeric_graph_id: &'static str,
    pub protected_graph_id: &'static str,
    pub method_id: &'static str,
    pub method_artifact_digest: Digest,
    pub kernel_id: &'static str,
    pub kernel_artifact_digest: Digest,
}

pub fn silu_q7_installed_contract() -> SiluQ7InstalledContract {
    SiluQ7InstalledContract {
        profile: SILU_Q7_EXPERIMENT_PROFILE,
        compiler_id: SILU_Q7_COMPILER_ID,
        compiler_version: env!("CARGO_PKG_VERSION"),
        compiler_artifact_digest: silu_q7_compiler_artifact_digest(),
        numeric_graph_id: SILU_Q7_NUMERIC_GRAPH_ID,
        protected_graph_id: SILU_Q7_PROTECTED_GRAPH_ID,
        method_id: SILU_Q7_METHOD_ID,
        method_artifact_digest: silu_q7_method_artifact_digest(),
        kernel_id: SILU_Q7_KERNEL_DESCRIPTOR_ID,
        kernel_artifact_digest: silu_q7_kernel_artifact_digest(),
    }
}

pub fn silu_q7_method_artifact_digest() -> Digest {
    canonical_digest(
        "pllm.artifact.rust-source-set.v1",
        &[
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-garble/src/lib.rs"),
            ),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-garble/src/gated_multiply_q7.rs"),
            ),
        ],
    )
}

pub fn gated_multiply_q7_method_artifact_digest() -> Digest {
    silu_q7_method_artifact_digest()
}

pub fn gated_multiply_q7_kernel_artifact_digest() -> Digest {
    canonical_digest(
        "pllm.artifact.rust-source-set.v1",
        &[
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-core/src/activation.rs"),
            ),
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-core/src/fixed_point.rs"),
            ),
            gated_multiply_q7_method_artifact_digest(),
        ],
    )
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityLevel {
    ExecutableRegion,
    Primitive,
    Missing,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OperatorCoverage {
    pub operator: ModelOperator,
    pub occurrences: u64,
    pub level: CapabilityLevel,
    pub component: Option<String>,
    pub blocker: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecoderCoverageReport {
    pub schema_version: String,
    pub profile: String,
    pub model_config_digest: Digest,
    pub complete: bool,
    pub operators: Vec<OperatorCoverage>,
}

pub fn decoder_coverage(plan: &DecoderPlan, profile: &str) -> DecoderCoverageReport {
    let mut occurrences = BTreeMap::<ModelOperator, u64>::new();
    for operation in plan
        .prefill
        .operations
        .iter()
        .chain(&plan.decode.operations)
    {
        *occurrences.entry(operation.operator).or_default() += 1;
    }
    let reshape_executable = lower_model_reshape_regions(plan, DecoderMode::Prefill).is_ok()
        && lower_model_reshape_regions(plan, DecoderMode::Decode).is_ok();
    let residual_executable = lower_model_residual_regions(plan, DecoderMode::Prefill).is_ok()
        && lower_model_residual_regions(plan, DecoderMode::Decode).is_ok();
    let output_head_executable = lower_model_output_head_regions(plan, DecoderMode::Prefill)
        .is_ok()
        && lower_model_output_head_regions(plan, DecoderMode::Decode).is_ok();
    let last_token_executable = lower_model_last_token_regions(plan, DecoderMode::Prefill).is_ok()
        && lower_model_last_token_regions(plan, DecoderMode::Decode).is_ok();
    let greedy_executable = lower_model_greedy_token_selection_regions(plan, DecoderMode::Prefill)
        .is_ok()
        && lower_model_greedy_token_selection_regions(plan, DecoderMode::Decode).is_ok();
    let feedback_executable = lower_model_token_feedback_regions(plan, DecoderMode::Prefill)
        .is_ok()
        && lower_model_token_feedback_regions(plan, DecoderMode::Decode).is_ok();
    let softmax_executable = lower_model_softmax_q30_regions(plan, DecoderMode::Prefill).is_ok()
        && lower_model_softmax_q30_regions(plan, DecoderMode::Decode).is_ok();
    let attention_values_executable =
        lower_model_attention_values_q10_regions(plan, DecoderMode::Prefill).is_ok()
            && lower_model_attention_values_q10_regions(plan, DecoderMode::Decode).is_ok();
    let attention_scores_executable =
        lower_model_attention_scores_q20_regions(plan, DecoderMode::Prefill).is_ok()
            && lower_model_attention_scores_q20_regions(plan, DecoderMode::Decode).is_ok();
    let rms_norm_reference = lower_rms_norm_f32_direct_regions(plan, DecoderMode::Prefill).is_ok()
        && lower_rms_norm_f32_direct_regions(plan, DecoderMode::Decode).is_ok();
    let provenance_q10_coverage = provenance_primitives::model_provenance_q10_coverage(plan);
    let operators = occurrences
        .into_iter()
        .map(|(operator, occurrences)| {
            let executable = operator == ModelOperator::Linear
                || (operator == ModelOperator::Reshape && reshape_executable)
                || (operator == ModelOperator::ResidualAdd && residual_executable)
                || (operator == ModelOperator::LastToken && last_token_executable)
                || (operator == ModelOperator::OutputHead && output_head_executable)
                || (operator == ModelOperator::GreedyTokenSelection && greedy_executable)
                || (operator == ModelOperator::TokenFeedback && feedback_executable)
                || (operator == ModelOperator::Softmax && softmax_executable)
                || (operator == ModelOperator::AttentionValues && attention_values_executable)
                || (matches!(
                    operator,
                    ModelOperator::AttentionScores
                        | ModelOperator::AttentionScale
                        | ModelOperator::CausalMask
                ) && attention_scores_executable);
            let descriptor_coverage = match operator {
                ModelOperator::RotaryEmbedding => Some(&provenance_q10_coverage.rope),
                ModelOperator::KvCacheAppend => Some(&provenance_q10_coverage.cache_append),
                ModelOperator::CacheSuffix => Some(&provenance_q10_coverage.cache_view),
                _ => None,
            };
            let descriptor_primitive = descriptor_coverage.is_some_and(Result::is_ok);
            let primitive = descriptor_primitive
                || matches!(
                operator,
                ModelOperator::TokenLookup
                    | ModelOperator::Reshape
                    | ModelOperator::Linear
                    | ModelOperator::AttentionScores
                    | ModelOperator::AttentionScale
                    | ModelOperator::CausalMask
                    | ModelOperator::AttentionValues
                    | ModelOperator::ResidualAdd
                    | ModelOperator::Silu
                    | ModelOperator::Multiply
                    | ModelOperator::LastToken
                    | ModelOperator::OutputHead
            );
            OperatorCoverage {
                operator,
                occurrences,
                level: if executable {
                    CapabilityLevel::ExecutableRegion
                } else if primitive || (operator == ModelOperator::RmsNorm && rms_norm_reference) {
                    CapabilityLevel::Primitive
                } else {
                    CapabilityLevel::Missing
                },
                component: if operator == ModelOperator::Linear {
                    Some("pllm/compiler-wrap32@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::Reshape && executable {
                    Some("pllm/compiler-layout@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::ResidualAdd && executable {
                    Some("pllm/core-tensor@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::LastToken && executable {
                    Some("pllm/compiler-layout@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::GreedyTokenSelection && executable {
                    Some("pllm/core-greedy-signed-wrap32@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::TokenFeedback && executable {
                    Some("pllm/core-token-feedback@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::Softmax && executable {
                    Some("pllm/client-softmax-q30@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::AttentionValues && executable {
                    Some("pllm/client-attention-values-q10@0.1.0-alpha.1".to_owned())
                } else if matches!(
                    operator,
                    ModelOperator::AttentionScores
                        | ModelOperator::AttentionScale
                        | ModelOperator::CausalMask
                ) && executable
                {
                    Some("pllm/client-attention-scores-q20@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::OutputHead && executable {
                    Some("pllm/compiler-wrap32@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::Silu {
                    Some("pllm/agc-silu-q7@0.1.0-alpha.1-experimental".to_owned())
                } else if operator == ModelOperator::RmsNorm && rms_norm_reference {
                    Some("pllm/core-rms-norm-f32-reference@0.1.0-alpha.1".to_owned())
                } else if operator == ModelOperator::Multiply {
                    Some(
                        "pllm/agc-gated-multiply-q7@0.1.0-alpha.1-experimental".to_owned(),
                    )
                } else if descriptor_primitive {
                    Some("pllm/core-provenance-primitives@0.1.0-alpha.1-reference".to_owned())
                } else if primitive {
                    Some("pllm/agc-project@0.1.0-alpha.1-reference".to_owned())
                } else {
                    None
                },
                blocker: if operator == ModelOperator::Linear {
                    "single semantic linear regions execute, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::Reshape && executable {
                    "semantic layout transformations execute, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::ResidualAdd && executable {
                    "semantic residual additions execute, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::LastToken {
                    if executable {
                        "semantic last-valid/physical-last selections execute, but whole-decoder scheduling is unavailable"
                            .to_owned()
                    } else {
                        "length-aware selection for last token is not executable; whole-decoder scheduling is unavailable"
                            .to_owned()
                    }
                } else if operator == ModelOperator::GreedyTokenSelection && executable {
                    "signed wrap32 greedy token selection executes with lowest-index tie-breaking, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::TokenFeedback && executable {
                    "token-id feedback layout executes, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::Softmax && executable {
                    "client-local Q20-to-Q30 softmax executes with exact row sums, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::AttentionValues && executable {
                    "client-local Q30-probability by Q10-value attention contraction executes, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::AttentionScores && executable {
                    "client-local Q10 attention scoring executes with exact Q20 scaling and masking, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::AttentionScale && executable {
                    "attention scaling is fused into the plan-bound Q20 score region, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::CausalMask && executable {
                    "causal and validity masking is fused into the plan-bound Q20 score region, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::OutputHead && executable {
                    "semantic output heads execute, but whole-decoder scheduling is unavailable"
                        .to_owned()
                } else if operator == ModelOperator::Silu {
                    "bounded one-use Q7 SiLU regions execute, but tensor composition and whole-decoder scheduling are unavailable"
                        .to_owned()
                } else if operator == ModelOperator::RmsNorm && rms_norm_reference {
                    "a plan-bound clear FP32 reference region executes, but no protected numeric decomposition or distributed executor is available"
                        .to_owned()
                } else if operator == ModelOperator::Multiply {
                    "gated-MLP SiLU and multiplication compose without decoding through scalar, four-lane, or resource-bounded chunked independent-lane harnesses; complete numeric scheduling and other multiplication contracts are unavailable"
                        .to_owned()
                } else if let Some(Err(error)) = descriptor_coverage {
                    format!(
                        "exact provenance-bound Q10 reference primitive is unavailable: {error}"
                    )
                } else if primitive {
                    "reference primitive exists but no compiled distributed executor is available"
                        .to_owned()
                } else {
                    "no reviewed numeric decomposition or executable component is available"
                        .to_owned()
                },
            }
        })
        .collect();
    DecoderCoverageReport {
        schema_version: "pllm.decoder_coverage_report.v1".to_owned(),
        profile: profile.to_owned(),
        model_config_digest: plan.config_digest.clone(),
        complete: false,
        operators,
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    Public,
    ClientPlaintext,
    MaskedRing,
    ArithmeticLabel,
    BooleanLabel,
    AdditiveShare,
    Ciphertext,
    AttestedPlaintext,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericType {
    Wrap32,
    SignedFixed16,
    SignedFixedQ7,
    SignedFixedQ10,
    SignedFixedQ20,
    UnsignedFixedQ30,
    TokenIdU32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorType {
    pub numeric: NumericType,
    pub shape: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operator {
    Linear,
    Silu,
    Conversion,
    Unsupported,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalOperation {
    pub id: String,
    pub operator: Operator,
    pub output: TensorType,
    pub input_representation: Representation,
    pub output_representation: Representation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelLinearRegion {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub weight_id: String,
    pub bias_id: Option<String>,
    pub input: TensorType,
    pub operation: LogicalOperation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelReshapeLayout {
    BatchHeadsSequenceFeature,
    BatchSequenceHidden,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelReshapeRegion {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub layout: ModelReshapeLayout,
    pub input: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelResidualRegion {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_ids: [String; 2],
    pub input: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelOutputHeadRegion {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub weight_id: String,
    pub input: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLastTokenSelection {
    PhysicalLast,
    LastValid,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelLastTokenRegion {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub sequence_lengths_input_id: Option<String>,
    pub selection: ModelLastTokenSelection,
    pub axis: usize,
    pub input: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelGreedyTokenSelectionRegion {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub input: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelTokenFeedbackRegion {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub input: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelSoftmaxQ30Region {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub axis: usize,
    pub numeric_profile: String,
    pub input: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelAttentionValuesQ10Region {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub probabilities_input_id: String,
    pub values_input_id: String,
    pub numeric_profile: String,
    pub layout: pllm_core::AttentionValueQ10Layout,
    pub probabilities: TensorType,
    pub values: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelAttentionScoresQ20Region {
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub score_operation_id: String,
    pub scale_operation_id: String,
    pub mask_operation_id: String,
    pub query_input_id: String,
    pub key_input_id: String,
    pub selected_positions_input_id: Option<String>,
    pub numeric_profile: String,
    pub layout: pllm_core::AttentionScoreQ20Layout,
    pub query: TensorType,
    pub key: TensorType,
    pub output: TensorType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RmsNormWeightPolicy {
    Direct,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RmsNormReductionOrder {
    LastAxisScalarLeftToRight,
}

/// Plan-bound clear reference for direct-weight FP32 RMSNorm semantics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelRmsNormF32Region {
    pub schema_version: String,
    pub model_plan_digest: Digest,
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub weight_id: String,
    pub epsilon: String,
    pub shape: Vec<u64>,
    pub kernel_artifact_digest: Digest,
    pub numeric_profile: String,
    pub weight_policy: RmsNormWeightPolicy,
    pub reduction_order: RmsNormReductionOrder,
}

/// Plan-bound bounded-integer profile for direct-weight RMSNorm semantics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelRmsNormQ10DirectRegion {
    pub schema_version: String,
    pub model_plan_digest: Digest,
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub weight_id: String,
    pub epsilon_numerator: u64,
    pub epsilon_denominator: u64,
    pub shape: Vec<u64>,
    pub kernel_artifact_digest: Digest,
    pub numeric_profile: String,
    pub input_fractional_bits: u8,
    pub weight_fractional_bits: u8,
    pub reciprocal_fractional_bits: u8,
    pub output_fractional_bits: u8,
    pub maximum_width: usize,
    pub maximum_encoded_error: u32,
    pub weight_policy: RmsNormWeightPolicy,
    pub reduction_order: RmsNormReductionOrder,
    pub rounding: FixedPointRounding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedPointRounding {
    TiesToEven,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedPointRangePolicy {
    RejectOutsideUnitInterval,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Q14ToQ7RescaleRegion {
    pub schema_version: String,
    pub operation_id: String,
    pub source_operation_id: String,
    pub target_operation_id: String,
    pub target_input_index: u32,
    pub numeric_profile: String,
    pub input: TensorType,
    pub output: TensorType,
    pub input_fractional_bits: u8,
    pub output_fractional_bits: u8,
    pub divisor: u32,
    pub rounding: FixedPointRounding,
    pub range_policy: FixedPointRangePolicy,
}

/// Exact semantic and installed-component binding for one gated Q7 MLP region.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelGatedMultiplyQ7Region {
    pub schema_version: String,
    pub profile: String,
    pub model_plan_digest: Digest,
    pub mode: DecoderMode,
    pub layer: u64,
    pub gate_linear_operation_id: String,
    pub gate_rescale: Q14ToQ7RescaleRegion,
    pub silu_operation_id: String,
    pub up_linear_operation_id: String,
    pub up_rescale: Q14ToQ7RescaleRegion,
    pub multiply_operation_id: String,
    pub input: TensorType,
    pub output: TensorType,
    pub numeric_graph_id: String,
    pub protected_graph_id: String,
    pub method_component_id: String,
    pub schedule_component_id: String,
    pub max_tensor_elements: usize,
    pub method: VersionedArtifact,
    pub kernel: VersionedArtifact,
    pub compiler: VersionedArtifact,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityProperties {
    pub online_parties: u16,
    pub needs_online_preparation: bool,
    pub needs_client_weights: bool,
    pub uses_he: bool,
    pub experimental: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MethodDescriptor {
    pub id: String,
    pub version: String,
    pub operator: Operator,
    pub input_representation: Representation,
    pub output_representation: Representation,
    pub properties: SecurityProperties,
    pub artifact_digest: Digest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelImplementation {
    PllmCoreMatrixWrap32,
    PllmGarbleSiluQuadraticQ7,
    ExplicitRepresentationConversion,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelDescriptor {
    pub id: String,
    pub version: String,
    pub method_id: String,
    pub method_version: String,
    pub input_numeric: NumericType,
    pub output_numeric: NumericType,
    pub implementation: KernelImplementation,
    pub artifact_digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEvidence {
    pub method_id: String,
    pub method_version: String,
    pub method_artifact_digest: Digest,
    pub kernel_id: String,
    pub kernel_version: String,
    pub kernel_artifact_digest: Digest,
    pub assurance_results: Vec<EvidenceReference>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompileRequest {
    pub configuration_json: Vec<u8>,
    pub configuration_digest: Digest,
    pub context: LockedContext,
    pub privacy_contract: PrivacyContract,
    pub input: TensorType,
    pub input_representation: Representation,
    pub output: TensorType,
    pub output_representation: Representation,
    pub operations: Vec<LogicalOperation>,
    pub methods: Vec<MethodDescriptor>,
    pub kernels: Vec<KernelDescriptor>,
    pub assurance_results: Vec<AssuranceResult>,
    pub candidate_evidence: Vec<CandidateEvidence>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompileDocument {
    schema_version: String,
    configuration: ExperimentDocument,
    context: LockedContext,
    privacy_contract: PrivacyContract,
    input: TensorType,
    input_representation: Representation,
    output: TensorType,
    output_representation: Representation,
    operations: Vec<LogicalOperation>,
    methods: Vec<MethodDescriptor>,
    kernels: Vec<KernelDescriptor>,
    assurance_results: Vec<AssuranceResult>,
    candidate_evidence: Vec<CandidateEvidence>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentDocument {
    schema: String,
    name: String,
    pipeline: ExperimentPipeline,
    deployment: ExperimentDeployment,
    budget: ExperimentBudget,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentPipeline {
    profile: String,
    model: ExperimentModel,
    components: BTreeMap<String, ExperimentComponent>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentModel {
    source: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentComponent {
    component: String,
    params: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentDeployment {
    kind: String,
    root: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentBudget {
    requests: u64,
    max_input_tokens: u64,
    max_new_tokens: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedExperimentProfile {
    canonical_profile: Vec<u8>,
    configuration_digest: Digest,
    model: String,
}

impl ResolvedExperimentProfile {
    pub fn canonical_profile(&self) -> &[u8] {
        &self.canonical_profile
    }

    pub fn configuration_digest(&self) -> &Digest {
        &self.configuration_digest
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegionGraphDigests {
    pub semantic_graph: Digest,
    pub numeric_graph: Digest,
    pub protected_graph: Digest,
}

#[derive(Serialize)]
struct SemanticOperation<'a> {
    id: &'a str,
    operator: Operator,
    output_shape: &'a [u64],
}

#[derive(Serialize)]
struct SemanticIntent<'a> {
    input_shape: &'a [u64],
    output_shape: &'a [u64],
    operations: Vec<SemanticOperation<'a>>,
}

#[derive(Serialize)]
struct NumericOperation {
    output_numeric: NumericType,
}

#[derive(Serialize)]
struct NumericIntent<'a> {
    semantic_graph_digest: &'a Digest,
    input_numeric: NumericType,
    output_numeric: NumericType,
    operations: Vec<NumericOperation>,
}

#[derive(Serialize)]
struct ProtectedOperation {
    input_representation: Representation,
    output_representation: Representation,
}

#[derive(Serialize)]
struct ProtectedIntent<'a> {
    numeric_graph_digest: &'a Digest,
    input_representation: Representation,
    output_representation: Representation,
    operations: Vec<ProtectedOperation>,
}

pub fn region_graph_digests(
    input: &TensorType,
    input_representation: Representation,
    output: &TensorType,
    output_representation: Representation,
    operations: &[LogicalOperation],
) -> RegionGraphDigests {
    let semantic_intent = SemanticIntent {
        input_shape: &input.shape,
        output_shape: &output.shape,
        operations: operations
            .iter()
            .map(|operation| SemanticOperation {
                id: &operation.id,
                operator: operation.operator,
                output_shape: &operation.output.shape,
            })
            .collect(),
    };
    let semantic_graph = pllm_types::canonical_digest("pllm.semantic_region.v1", &semantic_intent);
    let numeric_intent = NumericIntent {
        semantic_graph_digest: &semantic_graph,
        input_numeric: input.numeric,
        output_numeric: output.numeric,
        operations: operations
            .iter()
            .map(|operation| NumericOperation {
                output_numeric: operation.output.numeric,
            })
            .collect(),
    };
    let numeric_graph = pllm_types::canonical_digest("pllm.numeric_region.v1", &numeric_intent);
    let protected_intent = ProtectedIntent {
        numeric_graph_digest: &numeric_graph,
        input_representation,
        output_representation,
        operations: operations
            .iter()
            .map(|operation| ProtectedOperation {
                input_representation: operation.input_representation,
                output_representation: operation.output_representation,
            })
            .collect(),
    };
    let protected_graph =
        pllm_types::canonical_digest("pllm.protected_region.v1", &protected_intent);
    RegionGraphDigests {
        semantic_graph,
        numeric_graph,
        protected_graph,
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegionStep {
    pub operation_id: String,
    pub operator: Operator,
    pub input: TensorType,
    pub output: TensorType,
    pub input_representation: Representation,
    pub output_representation: Representation,
    pub method: MethodDescriptor,
    pub kernel: KernelDescriptor,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegionProgram {
    pub schema_version: String,
    pub logical_plan_digest: Digest,
    pub execution_role: String,
    pub input: TensorType,
    pub input_representation: Representation,
    pub output: TensorType,
    pub output_representation: Representation,
    pub steps: Vec<RegionStep>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPlan {
    pub logical: LogicalPlan,
    pub execution: ExecutionPlan,
    pub lock: PlanLock,
    pub region_program: RegionProgram,
    pub configuration_json: Vec<u8>,
    pub locked_context: LockedContext,
    pub privacy_contract: PrivacyContract,
    pub assurance_results: Vec<AssuranceResult>,
    pub candidate_evidence: Vec<CandidateEvidence>,
}

impl CompiledPlan {
    pub fn logical_json(&self) -> Vec<u8> {
        pllm_types::logical_plan_bytes(&self.logical)
    }

    pub fn execution_json(&self) -> Vec<u8> {
        pllm_types::execution_plan_bytes(&self.execution)
    }

    pub fn lock_json(&self) -> Vec<u8> {
        plan_lock_bytes(&self.lock)
    }

    pub fn region_program_json(&self) -> Vec<u8> {
        region_program_bytes(&self.region_program)
    }

    pub fn verify(&self) -> Result<(), String> {
        verify_compiled_plan(self)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum DiagnosticCode {
    #[serde(rename = "E_CLIENT_WEIGHTS")]
    ClientWeights,
    #[serde(rename = "E_DUPLICATE_ID")]
    DuplicateId,
    #[serde(rename = "E_EXPERIMENTAL")]
    Experimental,
    #[serde(rename = "E_HE_FORBIDDEN")]
    HeForbidden,
    #[serde(rename = "E_INCOMPLETE_STEP")]
    IncompleteStep,
    #[serde(rename = "E_INVALID_ASSURANCE")]
    InvalidAssurance,
    #[serde(rename = "E_INVALID_CONTEXT")]
    InvalidContext,
    #[serde(rename = "E_INVALID_CONTRACT")]
    InvalidContract,
    #[serde(rename = "E_INVALID_DOCUMENT")]
    InvalidDocument,
    #[serde(rename = "E_INVALID_ID")]
    InvalidId,
    #[serde(rename = "E_INVALID_TENSOR")]
    InvalidTensor,
    #[serde(rename = "E_INVALID_VERSION")]
    InvalidVersion,
    #[serde(rename = "E_MISSING_CLAIM")]
    MissingClaim,
    #[serde(rename = "E_NOT_IMPLEMENTED")]
    NotImplemented,
    #[serde(rename = "E_ONLINE_PREPARATION")]
    OnlinePreparation,
    #[serde(rename = "E_PARTY_MISMATCH")]
    PartyMismatch,
    #[serde(rename = "E_REPRESENTATION")]
    Representation,
    #[serde(rename = "E_UNACCEPTED_OUTCOME")]
    UnacceptedOutcome,
    #[serde(rename = "E_UNACCEPTED_REFINEMENT")]
    UnacceptedRefinement,
    #[serde(rename = "E_UNKNOWN_EVIDENCE")]
    UnknownEvidence,
    #[serde(rename = "E_UNKNOWN_METHOD")]
    UnknownMethod,
    #[serde(rename = "E_UNSUPPORTED_OPERATOR")]
    UnsupportedOperator,
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let encoded = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        formatter.write_str(encoded.trim_matches('"'))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub subject_id: String,
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} [{}]: {}",
            self.code, self.subject_id, self.message
        )
    }
}

impl std::error::Error for Diagnostic {}

fn diagnostic(code: DiagnosticCode, subject_id: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic {
        code,
        subject_id: subject_id.to_owned(),
        message: message.into(),
    }
}

fn sorted_diagnostics(mut diagnostics: Vec<Diagnostic>) -> Vec<Diagnostic> {
    diagnostics.sort();
    diagnostics.dedup();
    diagnostics
}

pub fn diagnostics_json(diagnostics: &[Diagnostic]) -> Vec<u8> {
    canonical_bytes(&sorted_diagnostics(diagnostics.to_vec()))
}

fn document_error(message: impl Into<String>) -> Vec<Diagnostic> {
    vec![diagnostic(
        DiagnosticCode::InvalidDocument,
        "compile_document",
        message,
    )]
}

fn validate_experiment(document: &ExperimentDocument) -> Result<(), String> {
    if document.schema != "pllm.experiment.v1" {
        return Err("configuration schema must be pllm.experiment.v1".into());
    }
    if document.name.is_empty()
        || document.pipeline.profile.is_empty()
        || document.pipeline.model.source.is_empty()
        || document.deployment.kind != "local"
        || document.deployment.root.is_empty()
        || document.budget.requests == 0
        || document.budget.max_input_tokens == 0
        || document.budget.max_new_tokens == 0
    {
        return Err("embedded Experiment contains an invalid required value".into());
    }

    for (slot, component) in &document.pipeline.components {
        if slot.is_empty()
            || slot.contains("__")
            || component.component.is_empty()
            || component.params.keys().any(|key| {
                key.is_empty() || key == "component" || key == "params" || key.contains("__")
            })
        {
            return Err(format!("configuration component {slot} is malformed"));
        }
        if component.component == "pllm/cpu"
            && (component.params.len() != 1
                || component
                    .params
                    .get("threads")
                    .and_then(serde_json::Value::as_u64)
                    == Some(0)
                || component
                    .params
                    .get("threads")
                    .and_then(serde_json::Value::as_u64)
                    .is_none())
        {
            return Err(format!(
                "configuration component {slot} requires positive integer threads"
            ));
        }
        if matches!(
            component.component.as_str(),
            "pllm/masked-linear" | "pllm/model-aware-corrections" | "pllm/inference"
        ) && !component.params.is_empty()
        {
            return Err(format!(
                "configuration component {slot} does not accept parameters"
            ));
        }
        if matches!(
            component.component.as_str(),
            GATED_MULTIPLY_Q7_BINARY_TABLE_COMPONENT_ID
                | GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID
                | GATED_MULTIPLY_Q7_SCALAR_SCHEDULE_COMPONENT_ID
        ) && !component.params.is_empty()
        {
            return Err(format!(
                "configuration component {slot} does not accept parameters"
            ));
        }
        if component.component == GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID {
            let max_elements = component
                .params
                .get("max_elements")
                .and_then(serde_json::Value::as_u64);
            if component.params.len() != 1 || !matches!(max_elements, Some(2..=4)) {
                return Err(format!(
                    "configuration component {slot} requires max_elements between 2 and 4"
                ));
            }
        }
        if component.component == GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID
        {
            let max_elements = component
                .params
                .get("max_elements")
                .and_then(serde_json::Value::as_u64);
            if component.params.len() != 1 || !matches!(max_elements, Some(5..=4_000_000)) {
                return Err(format!(
                    "configuration component {slot} requires max_elements between 5 and 4000000"
                ));
            }
        }
    }
    Ok(())
}

fn validate_baseline_components(document: &ExperimentDocument) -> Result<(), String> {
    for (slot, required) in [
        ("linear", "pllm/masked-linear"),
        ("preparation", "pllm/model-aware-corrections"),
        ("inference", "pllm/inference"),
    ] {
        let component = document
            .pipeline
            .components
            .get(slot)
            .ok_or_else(|| format!("baseline profile requires {slot} component {required}"))?;
        if component.component != required || !component.params.is_empty() {
            return Err(format!(
                "baseline profile requires {slot} component {required} with no parameters"
            ));
        }
    }
    let kernels = document
        .pipeline
        .components
        .get("kernels")
        .ok_or_else(|| "baseline profile requires kernels component pllm/cpu".to_string())?;
    if kernels.component != "pllm/cpu"
        || kernels.params.len() != 1
        || kernels
            .params
            .get("threads")
            .and_then(serde_json::Value::as_u64)
            .is_none_or(|threads| threads == 0)
    {
        return Err(
            "baseline profile requires kernels component pllm/cpu with positive integer threads"
                .into(),
        );
    }
    if document.pipeline.components.len() != 4 {
        return Err("baseline profile requires exactly linear, preparation, inference, and kernels components".into());
    }
    Ok(())
}

pub fn resolve_experiment(bytes: &[u8]) -> Result<ResolvedExperimentProfile, String> {
    let document: ExperimentDocument = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid Experiment document: {error}"))?;
    if canonical_bytes(&document) != bytes {
        return Err("Experiment must use canonical compact sorted JSON bytes".into());
    }
    validate_experiment(&document)?;
    if document.pipeline.profile != BASELINE_EXPERIMENT_PROFILE {
        return Err(format!(
            "unsupported Experiment profile {:?}",
            document.pipeline.profile
        ));
    }
    validate_baseline_components(&document)?;

    Ok(ResolvedExperimentProfile {
        canonical_profile: canonical_bytes(&document.pipeline),
        configuration_digest: configuration_digest_bytes(bytes),
        model: document.pipeline.model.source,
    })
}

fn request_from_document(document: CompileDocument) -> CompileRequest {
    let configuration_json = canonical_bytes(&document.configuration);
    let configuration_digest = configuration_digest_bytes(&configuration_json);
    CompileRequest {
        configuration_json,
        configuration_digest,
        context: document.context,
        privacy_contract: document.privacy_contract,
        input: document.input,
        input_representation: document.input_representation,
        output: document.output,
        output_representation: document.output_representation,
        operations: document.operations,
        methods: document.methods,
        kernels: document.kernels,
        assurance_results: document.assurance_results,
        candidate_evidence: document.candidate_evidence,
    }
}

fn sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn valid_named(value: &NamedDigest) -> bool {
    valid_identity(&value.id)
}

fn validate_configuration_json(
    bytes: &[u8],
    declared_digest: &Digest,
) -> Result<serde_json::Value, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| "configuration JSON is invalid")?;
    if canonical_bytes(&value) != bytes {
        return Err("configuration JSON is not canonical".into());
    }
    if value.get("schema").and_then(serde_json::Value::as_str) != Some("pllm.experiment.v1") {
        return Err("configuration schema must be pllm.experiment.v1".into());
    }
    if configuration_digest_bytes(bytes) != *declared_digest {
        return Err("configuration digest does not match canonical Experiment bytes".into());
    }
    Ok(value)
}

fn configuration_profile(value: &serde_json::Value) -> Option<&str> {
    value
        .pointer("/pipeline/profile")
        .or_else(|| value.get("profile"))
        .and_then(serde_json::Value::as_str)
}

fn validate_context(request: &CompileRequest) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let context = &request.context;
    match validate_configuration_json(&request.configuration_json, &request.configuration_digest) {
        Err(message) => diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContext,
            "configuration",
            message,
        )),
        Ok(configuration) => {
            if configuration_profile(&configuration) != Some(context.profile.as_str()) {
                diagnostics.push(diagnostic(
                    DiagnosticCode::InvalidContext,
                    "configuration.profile",
                    "configuration profile does not match locked context",
                ));
            }
        }
    }
    if context.schema_version != LOCKED_CONTEXT_SCHEMA_VERSION {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContext,
            "locked_context",
            "schema_version must be pllm.locked_context.v1",
        ));
    }
    let graphs = region_graph_digests(
        &request.input,
        request.input_representation,
        &request.output,
        request.output_representation,
        &request.operations,
    );
    if context.semantic_graph.digest != graphs.semantic_graph
        || context.numeric_graph.digest != graphs.numeric_graph
        || context.protected_graph.digest != graphs.protected_graph
    {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContext,
            "region_graphs",
            "semantic, numeric, or protected graph digest does not match region intent",
        ));
    }
    if !valid_identity(&context.profile)
        || !valid_named(&context.model)
        || !valid_named(&context.tokenizer)
        || !valid_named(&context.semantic_graph)
        || !valid_named(&context.numeric_graph)
        || !valid_named(&context.protected_graph)
        || !valid_named(&context.privacy_contract)
        || !valid_named(&context.workload)
        || !valid_named(&context.target)
        || !valid_named(&context.resource_forecast)
        || !valid_identity(&context.compiler.id)
        || !valid_identity(&context.compiler.version)
        || !valid_identity(&context.execution_role)
        || context
            .material_requests
            .iter()
            .any(|item| !valid_named(item))
        || context
            .static_role_plans
            .iter()
            .any(|item| !valid_identity(&item.role) || item.role == context.execution_role)
    {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContext,
            "locked_context",
            "locked context contains an empty or malformed identity",
        ));
    }
    let expected_static_roles: Vec<_> = context
        .roles
        .iter()
        .filter(|role| role.as_str() != context.execution_role)
        .map(String::as_str)
        .collect();
    let actual_static_roles: Vec<_> = context
        .static_role_plans
        .iter()
        .map(|plan| plan.role.as_str())
        .collect();
    if context.roles.is_empty()
        || !sorted_unique(&context.roles)
        || !context.roles.contains(&context.execution_role)
        || !sorted_unique(&context.static_role_plans)
        || actual_static_roles != expected_static_roles
    {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContext,
            "roles",
            "roles must be sorted and unique; static_role_plans must cover every non-execution role exactly once",
        ));
    }
    if !sorted_unique(&context.material_requests) {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContext,
            "material_requests",
            "material request references must be sorted and unique",
        ));
    }
    if request.privacy_contract.schema_version != PRIVACY_CONTRACT_SCHEMA_VERSION
        || request.privacy_contract.id != context.privacy_contract.id
        || privacy_contract_digest(&request.privacy_contract) != context.privacy_contract.digest
    {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContract,
            &request.privacy_contract.id,
            "privacy contract identity or digest does not match locked context",
        ));
    }
    if request.privacy_contract.online_parties == 0 {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContract,
            &request.privacy_contract.id,
            "online_parties must be positive",
        ));
    }
    let requirements = &request.privacy_contract.required_claims;
    if !requirements
        .windows(2)
        .all(|pair| pair[0].claim_id < pair[1].claim_id)
        || requirements.iter().any(|item| {
            !valid_identity(&item.claim_id)
                || item.accepted_outcomes.is_empty()
                || item.accepted_implementation_refinements.is_empty()
        })
    {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContract,
            &request.privacy_contract.id,
            "claim requirements must have sorted unique IDs and nonempty acceptance sets",
        ));
    }
    diagnostics
}

fn validate_assurance_result(result: &AssuranceResult) -> bool {
    result.schema_version == ASSURANCE_RESULT_SCHEMA_VERSION
        && valid_identity(&result.id)
        && valid_identity(&result.claim_id)
        && !result.scope.is_empty()
        && result.assumptions.iter().all(|value| !value.is_empty())
        && result.evidence_paths.iter().all(|value| !value.is_empty())
        && result.tool.as_deref().is_none_or(|value| !value.is_empty())
        && result
            .tool_version
            .as_deref()
            .is_none_or(|value| !value.is_empty())
}

fn validate_method_properties(
    privacy: &PrivacyContract,
    method: &MethodDescriptor,
) -> Vec<Diagnostic> {
    let id = format!("{}@{}", method.id, method.version);
    let properties = &method.properties;
    let mut diagnostics = Vec::new();
    if properties.online_parties != privacy.online_parties {
        diagnostics.push(diagnostic(
            DiagnosticCode::PartyMismatch,
            &id,
            "online party count differs from privacy contract",
        ));
    }
    if properties.needs_online_preparation && !privacy.allow_online_preparation {
        diagnostics.push(diagnostic(
            DiagnosticCode::OnlinePreparation,
            &id,
            "method requires forbidden online preparation",
        ));
    }
    if properties.needs_client_weights && !privacy.allow_client_weights {
        diagnostics.push(diagnostic(
            DiagnosticCode::ClientWeights,
            &id,
            "method requires forbidden client weights",
        ));
    }
    if properties.uses_he && !privacy.allow_he {
        diagnostics.push(diagnostic(
            DiagnosticCode::HeForbidden,
            &id,
            "method uses forbidden homomorphic encryption",
        ));
    }
    if properties.experimental && !privacy.allow_experimental {
        diagnostics.push(diagnostic(
            DiagnosticCode::Experimental,
            &id,
            "experimental method is not permitted",
        ));
    }
    diagnostics
}

fn implementation_matches(operator: Operator, implementation: KernelImplementation) -> bool {
    matches!(
        (operator, implementation),
        (Operator::Linear, KernelImplementation::PllmCoreMatrixWrap32)
            | (
                Operator::Silu,
                KernelImplementation::PllmGarbleSiluQuadraticQ7
            )
            | (
                Operator::Conversion,
                KernelImplementation::ExplicitRepresentationConversion
            )
    )
}

fn validate_silu_q7_descriptor(
    method: &MethodDescriptor,
    kernel: &KernelDescriptor,
) -> Option<Diagnostic> {
    if kernel.implementation != KernelImplementation::PllmGarbleSiluQuadraticQ7 {
        return None;
    }
    let expected_properties = SecurityProperties {
        online_parties: 1,
        needs_online_preparation: false,
        needs_client_weights: false,
        uses_he: false,
        experimental: true,
    };
    if method.id != SILU_Q7_METHOD_ID
        || method.version != "1"
        || method.artifact_digest != silu_q7_method_artifact_digest()
        || method.properties != expected_properties
        || method.input_representation != Representation::ArithmeticLabel
        || method.output_representation != Representation::ArithmeticLabel
        || kernel.id != SILU_Q7_KERNEL_DESCRIPTOR_ID
        || kernel.version != "1"
        || kernel.artifact_digest != silu_q7_kernel_artifact_digest()
    {
        return Some(diagnostic(
            DiagnosticCode::InvalidContract,
            &method.id,
            "Q7 SiLU method or kernel descriptor does not match the installed implementation",
        ));
    }
    None
}

fn validate_tensor(operation: &LogicalOperation, input: &TensorType) -> Result<(), Diagnostic> {
    match operation.operator {
        Operator::Linear => {
            if input.numeric != NumericType::Wrap32
                || operation.output.numeric != NumericType::Wrap32
                || input.shape.len() != 2
                || operation.output.shape.len() != 2
                || input.shape[0] != operation.output.shape[0]
                || input.shape[1] == 0
                || operation.output.shape[1] == 0
            {
                return Err(diagnostic(
                    DiagnosticCode::InvalidTensor,
                    &operation.id,
                    "linear wrap32 requires [batch,input] to [batch,output] with nonzero features",
                ));
            }
        }
        Operator::Silu => {
            if input.numeric != NumericType::SignedFixedQ7
                || operation.output.numeric != NumericType::SignedFixedQ7
                || input.shape != operation.output.shape
                || input.shape.is_empty()
                || input.shape.contains(&0)
                || operation.input_representation != Representation::ArithmeticLabel
                || operation.output_representation != Representation::ArithmeticLabel
            {
                return Err(diagnostic(
                    DiagnosticCode::InvalidTensor,
                    &operation.id,
                    "SiLU quadratic Q7 requires a nonempty shape-preserving signed_fixed_q7 tensor in arithmetic_label representation",
                ));
            }
            if input
                .shape
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .and_then(|count| usize::try_from(count).ok())
                .is_none()
            {
                return Err(diagnostic(
                    DiagnosticCode::InvalidTensor,
                    &operation.id,
                    "SiLU tensor element count exceeds the executor address space",
                ));
            }
            if input.shape.iter().product::<u64>() > SILU_Q7_MAX_TENSOR_ELEMENTS as u64 {
                return Err(diagnostic(
                    DiagnosticCode::InvalidTensor,
                    &operation.id,
                    "the bounded Q7 SiLU slice supports at most 128 elements",
                ));
            }
        }
        Operator::Conversion if input != &operation.output => {
            return Err(diagnostic(
                DiagnosticCode::InvalidTensor,
                &operation.id,
                "representation conversion must preserve tensor type",
            ));
        }
        Operator::Conversion | Operator::Unsupported => {}
    }
    Ok(())
}

fn encoded_enum<T: Serialize>(value: T) -> String {
    serde_json::to_string(&value)
        .expect("assurance enum is serializable")
        .trim_matches('"')
        .to_owned()
}

fn validate_candidate_assurance(
    privacy: &PrivacyContract,
    candidate_id: &str,
    artifact_digests: [&Digest; 2],
    references: &[EvidenceReference],
    results: &BTreeMap<&str, &AssuranceResult>,
) -> Vec<Diagnostic> {
    let requirements: BTreeMap<_, _> = privacy
        .required_claims
        .iter()
        .map(|item| (item.claim_id.as_str(), item))
        .collect();
    let mut seen_claims = BTreeSet::new();
    let mut diagnostics = Vec::new();
    for reference in references {
        let Some(result) = results.get(reference.id.as_str()) else {
            continue;
        };
        seen_claims.insert(result.claim_id.as_str());
        let Some(requirement) = requirements.get(result.claim_id.as_str()) else {
            diagnostics.push(diagnostic(
                DiagnosticCode::UnacceptedOutcome,
                candidate_id,
                format!(
                    "evidence {} addresses undeclared claim {}; declare it or remove reference",
                    result.id, result.claim_id
                ),
            ));
            continue;
        };
        if !requirement.accepted_outcomes.contains(&result.outcome) {
            diagnostics.push(diagnostic(
                DiagnosticCode::UnacceptedOutcome,
                candidate_id,
                format!(
                    "claim {} evidence {} has unaccepted outcome {}",
                    result.claim_id,
                    result.id,
                    encoded_enum(result.outcome)
                ),
            ));
        }
        if !requirement
            .accepted_implementation_refinements
            .contains(&result.implementation_refinement)
        {
            diagnostics.push(diagnostic(
                DiagnosticCode::UnacceptedRefinement,
                candidate_id,
                format!(
                    "claim {} evidence {} has unaccepted refinement {}",
                    result.claim_id,
                    result.id,
                    encoded_enum(result.implementation_refinement)
                ),
            ));
        }
        if result.implementation_refinement
            == pllm_types::ImplementationRefinement::ProvedForLockedCode
            && !result
                .code_digest
                .as_ref()
                .is_some_and(|digest| artifact_digests.contains(&digest))
        {
            diagnostics.push(diagnostic(
                DiagnosticCode::UnacceptedRefinement,
                candidate_id,
                format!(
                    "claim {} evidence {} does not bind selected method or kernel artifact",
                    result.claim_id, result.id
                ),
            ));
        }
    }
    for requirement in &privacy.required_claims {
        if !seen_claims.contains(requirement.claim_id.as_str()) {
            diagnostics.push(diagnostic(
                DiagnosticCode::MissingClaim,
                candidate_id,
                format!(
                    "required claim {} has no referenced assurance result",
                    requirement.claim_id
                ),
            ));
        }
    }
    diagnostics
}

#[derive(Clone, Copy)]
struct Candidate<'a> {
    method: &'a MethodDescriptor,
    kernel: &'a KernelDescriptor,
    evidence: &'a [EvidenceReference],
}

/// Compile one complete linear region from explicit locked inputs.
pub fn compile_document(bytes: &[u8]) -> Result<CompiledPlan, Vec<Diagnostic>> {
    let document: CompileDocument = serde_json::from_slice(bytes)
        .map_err(|error| document_error(format!("invalid compile document: {error}")))?;
    if document.schema_version != COMPILE_REQUEST_SCHEMA_VERSION {
        return Err(document_error(format!(
            "schema_version must be {COMPILE_REQUEST_SCHEMA_VERSION}"
        )));
    }
    validate_experiment(&document.configuration).map_err(document_error)?;
    if document.configuration.pipeline.profile == BASELINE_EXPERIMENT_PROFILE {
        validate_baseline_components(&document.configuration).map_err(document_error)?;
    }
    if canonical_bytes(&document) != bytes {
        return Err(document_error(
            "compile document must use canonical compact sorted JSON bytes",
        ));
    }
    compile(&request_from_document(document))
}

pub fn compile(request: &CompileRequest) -> Result<CompiledPlan, Vec<Diagnostic>> {
    let mut diagnostics = validate_context(request);
    let has_silu = request
        .operations
        .iter()
        .any(|operation| operation.operator == Operator::Silu);
    if has_silu && request.context.profile != SILU_Q7_EXPERIMENT_PROFILE {
        diagnostics.push(diagnostic(
            DiagnosticCode::InvalidContext,
            "configuration.profile",
            format!("Q7 SiLU requires profile {SILU_Q7_EXPERIMENT_PROFILE}"),
        ));
    }
    let mut operation_ids = BTreeMap::new();
    let mut method_ids = BTreeMap::new();
    let mut kernel_ids = BTreeMap::new();
    for operation in &request.operations {
        *operation_ids.entry(operation.id.as_str()).or_insert(0usize) += 1;
    }
    for method in &request.methods {
        *method_ids
            .entry((method.id.as_str(), method.version.as_str()))
            .or_insert(0usize) += 1;
    }
    for kernel in &request.kernels {
        *kernel_ids
            .entry((kernel.id.as_str(), kernel.version.as_str()))
            .or_insert(0usize) += 1;
    }
    if request
        .operations
        .iter()
        .any(|operation| operation.operator == Operator::Silu)
        && request.operations.len() != 1
    {
        diagnostics.push(diagnostic(
            DiagnosticCode::NotImplemented,
            "region",
            "the Q7 SiLU executor supports only a singleton region",
        ));
    }
    for (namespace, id, version, count) in operation_ids
        .into_iter()
        .map(|(id, count)| ("operation", id, None, count))
        .chain(
            method_ids
                .into_iter()
                .map(|((id, version), count)| ("method", id, Some(version), count)),
        )
        .chain(
            kernel_ids
                .into_iter()
                .map(|((id, version), count)| ("kernel", id, Some(version), count)),
        )
    {
        if !valid_identity(id) {
            diagnostics.push(diagnostic(
                DiagnosticCode::InvalidId,
                id,
                "identity must be nonempty portable ASCII",
            ));
        }
        if version.is_some_and(|value| !valid_identity(value)) {
            diagnostics.push(diagnostic(
                DiagnosticCode::InvalidVersion,
                id,
                "version must be nonempty portable ASCII",
            ));
        }
        if count > 1 {
            diagnostics.push(diagnostic(
                DiagnosticCode::DuplicateId,
                &version.map_or_else(|| id.to_owned(), |value| format!("{id}@{value}")),
                format!("{namespace} identity occurs more than once"),
            ));
        }
    }

    let methods: BTreeSet<_> = request
        .methods
        .iter()
        .map(|method| (method.id.as_str(), method.version.as_str()))
        .collect();
    let kernels: BTreeSet<_> = request
        .kernels
        .iter()
        .map(|kernel| (kernel.id.as_str(), kernel.version.as_str()))
        .collect();
    for kernel in &request.kernels {
        if !methods.contains(&(kernel.method_id.as_str(), kernel.method_version.as_str())) {
            diagnostics.push(diagnostic(
                DiagnosticCode::UnknownMethod,
                &format!("{}@{}", kernel.id, kernel.version),
                "kernel references unknown method identity",
            ));
        }
    }

    let mut assurance_results = BTreeMap::new();
    for result in &request.assurance_results {
        if !validate_assurance_result(result) {
            diagnostics.push(diagnostic(
                DiagnosticCode::InvalidAssurance,
                &result.id,
                "assurance result does not satisfy pllm.assurance_result.v1",
            ));
        }
        if assurance_results
            .insert(result.id.as_str(), result)
            .is_some()
        {
            diagnostics.push(diagnostic(
                DiagnosticCode::DuplicateId,
                &result.id,
                "assurance result identity occurs more than once",
            ));
        }
    }
    let mut candidate_evidence = BTreeMap::new();
    for binding in &request.candidate_evidence {
        let key = (
            binding.method_id.as_str(),
            binding.method_version.as_str(),
            &binding.method_artifact_digest,
            binding.kernel_id.as_str(),
            binding.kernel_version.as_str(),
            &binding.kernel_artifact_digest,
        );
        let id = format!("{}@{}/{}@{}", key.0, key.1, key.3, key.4);
        if candidate_evidence.insert(key, binding).is_some() {
            diagnostics.push(diagnostic(
                DiagnosticCode::DuplicateId,
                &id,
                "candidate evidence binding occurs more than once",
            ));
        }
        if !methods.contains(&(key.0, key.1)) || !kernels.contains(&(key.3, key.4)) {
            diagnostics.push(diagnostic(
                DiagnosticCode::InvalidAssurance,
                &id,
                "candidate evidence references unknown method or kernel",
            ));
        } else if request
            .methods
            .iter()
            .find(|method| method.id == key.0 && method.version == key.1)
            .is_none_or(|method| method.artifact_digest != *key.2)
            || request
                .kernels
                .iter()
                .find(|kernel| kernel.id == key.3 && kernel.version == key.4)
                .is_none_or(|kernel| kernel.artifact_digest != *key.5)
        {
            diagnostics.push(diagnostic(
                DiagnosticCode::InvalidAssurance,
                &id,
                "candidate evidence artifact digest does not match method or kernel",
            ));
        }
        let mut seen = BTreeSet::new();
        for reference in &binding.assurance_results {
            if !seen.insert(reference.id.as_str()) {
                diagnostics.push(diagnostic(
                    DiagnosticCode::DuplicateId,
                    &id,
                    "candidate references one assurance result more than once",
                ));
            }
            let Some(result) = assurance_results.get(reference.id.as_str()) else {
                diagnostics.push(diagnostic(
                    DiagnosticCode::UnknownEvidence,
                    &id,
                    format!("assurance result {} is absent", reference.id),
                ));
                continue;
            };
            if assurance_result_digest(result) != reference.digest {
                diagnostics.push(diagnostic(
                    DiagnosticCode::InvalidAssurance,
                    &id,
                    format!("assurance result {} digest mismatch", reference.id),
                ));
            }
        }
    }

    if request.operations.is_empty() {
        diagnostics.push(diagnostic(
            DiagnosticCode::IncompleteStep,
            "compile_request",
            "region must contain at least one operation",
        ));
    }
    let mut tensor = &request.input;
    let mut representation = request.input_representation;
    for operation in &request.operations {
        if operation.operator == Operator::Unsupported {
            diagnostics.push(diagnostic(
                DiagnosticCode::UnsupportedOperator,
                &operation.id,
                "operator is unsupported",
            ));
        }
        if operation.input_representation != representation {
            diagnostics.push(diagnostic(
                DiagnosticCode::Representation,
                &operation.id,
                "input representation does not match preceding output",
            ));
        }
        if let Err(error) = validate_tensor(operation, tensor) {
            diagnostics.push(error);
        }
        tensor = &operation.output;
        representation = operation.output_representation;
    }
    if !request.operations.is_empty()
        && (tensor != &request.output || representation != request.output_representation)
    {
        diagnostics.push(diagnostic(
            DiagnosticCode::IncompleteStep,
            "compile_request",
            "final operation does not produce declared region output",
        ));
    }
    if has_silu {
        if request.context.numeric_graph.id != SILU_Q7_NUMERIC_GRAPH_ID {
            diagnostics.push(diagnostic(
                DiagnosticCode::InvalidContext,
                "numeric_graph",
                "Q7 SiLU requires its exact installed numeric graph identity",
            ));
        }
        if request.context.protected_graph.id != SILU_Q7_PROTECTED_GRAPH_ID {
            diagnostics.push(diagnostic(
                DiagnosticCode::InvalidContext,
                "protected_graph",
                "Q7 SiLU requires its exact installed protected graph identity",
            ));
        }
        if request.context.compiler.id != SILU_Q7_COMPILER_ID
            || request.context.compiler.version != env!("CARGO_PKG_VERSION")
            || request.context.compiler.digest != silu_q7_compiler_artifact_digest()
        {
            diagnostics.push(diagnostic(
                DiagnosticCode::InvalidContext,
                "compiler",
                "Q7 SiLU requires the exact installed compiler artifact",
            ));
        }
    }
    if !diagnostics.is_empty() {
        return Err(sorted_diagnostics(diagnostics));
    }

    let logical = LogicalPlan {
        schema_version: LOGICAL_PLAN_SCHEMA_VERSION.into(),
        configuration_digest: request.configuration_digest.clone(),
        profile: request.context.profile.clone(),
        model: request.context.model.clone(),
        tokenizer: request.context.tokenizer.clone(),
        semantic_graph: request.context.semantic_graph.clone(),
        numeric_graph: request.context.numeric_graph.clone(),
        protected_graph: request.context.protected_graph.clone(),
        roles: request.context.roles.clone(),
        privacy_contract: request.context.privacy_contract.clone(),
        workload: request.context.workload.clone(),
        required_claim_ids: request
            .privacy_contract
            .required_claims
            .iter()
            .map(|item| item.claim_id.clone())
            .collect(),
    };
    let logical_digest = logical_plan_digest(&logical);

    let mut input = request.input.clone();
    let mut selected = Vec::with_capacity(request.operations.len());
    for operation in &request.operations {
        let mut candidates = Vec::new();
        let mut candidate_diagnostics = Vec::new();
        let matching_methods: Vec<_> = request
            .methods
            .iter()
            .filter(|method| method.operator == operation.operator)
            .collect();
        for method in &matching_methods {
            let property_errors = validate_method_properties(&request.privacy_contract, method);
            if !property_errors.is_empty() {
                candidate_diagnostics.extend(property_errors);
                continue;
            }
            let mut installed = false;
            for kernel in request.kernels.iter().filter(|kernel| {
                kernel.method_id == method.id
                    && kernel.method_version == method.version
                    && kernel.input_numeric == input.numeric
                    && kernel.output_numeric == operation.output.numeric
                    && implementation_matches(operation.operator, kernel.implementation)
            }) {
                installed = true;
                if let Some(error) = validate_silu_q7_descriptor(method, kernel) {
                    candidate_diagnostics.push(error);
                    continue;
                }
                let key = (
                    method.id.as_str(),
                    method.version.as_str(),
                    &method.artifact_digest,
                    kernel.id.as_str(),
                    kernel.version.as_str(),
                    &kernel.artifact_digest,
                );
                let evidence = candidate_evidence
                    .get(&key)
                    .map_or(&[][..], |binding| binding.assurance_results.as_slice());
                let id = format!("{}@{}/{}@{}", key.0, key.1, key.3, key.4);
                let assurance_errors = validate_candidate_assurance(
                    &request.privacy_contract,
                    &id,
                    [&method.artifact_digest, &kernel.artifact_digest],
                    evidence,
                    &assurance_results,
                );
                if assurance_errors.is_empty() {
                    candidates.push(Candidate {
                        method,
                        kernel,
                        evidence,
                    });
                } else {
                    candidate_diagnostics.extend(assurance_errors);
                }
            }
            if !installed {
                candidate_diagnostics.push(diagnostic(
                    DiagnosticCode::NotImplemented,
                    &format!("{}@{}", method.id, method.version),
                    "method has no compatible installed kernel",
                ));
            }
        }
        if matching_methods.is_empty() {
            candidate_diagnostics.push(diagnostic(
                DiagnosticCode::NotImplemented,
                &operation.id,
                "operator has no declared method",
            ));
        }
        let had_candidate = !candidates.is_empty();
        candidates.retain(|candidate| {
            candidate.method.input_representation == operation.input_representation
                && candidate.method.output_representation == operation.output_representation
        });
        if had_candidate && candidates.is_empty() {
            candidate_diagnostics.push(diagnostic(
                DiagnosticCode::Representation,
                &operation.id,
                "no candidate accepts required representations; add explicit conversion",
            ));
        }
        candidates.sort_by(|left, right| {
            (
                &left.method.id,
                &left.method.version,
                &left.kernel.id,
                &left.kernel.version,
            )
                .cmp(&(
                    &right.method.id,
                    &right.method.version,
                    &right.kernel.id,
                    &right.kernel.version,
                ))
        });
        let Some(candidate) = candidates.first().copied() else {
            diagnostics.extend(candidate_diagnostics);
            input = operation.output.clone();
            continue;
        };
        selected.push(candidate);
        input = operation.output.clone();
    }
    if selected.len() != request.operations.len() || !diagnostics.is_empty() {
        return Err(sorted_diagnostics(diagnostics));
    }

    let mut tensor = request.input.clone();
    let mut region_steps = Vec::with_capacity(selected.len());
    let mut components = Vec::with_capacity(selected.len() * 2);
    let mut conversions = Vec::new();
    let mut evidence_references = Vec::new();
    let mut selected_bindings = Vec::new();
    for (operation, candidate) in request.operations.iter().zip(selected) {
        region_steps.push(RegionStep {
            operation_id: operation.id.clone(),
            operator: operation.operator,
            input: tensor.clone(),
            output: operation.output.clone(),
            input_representation: operation.input_representation,
            output_representation: operation.output_representation,
            method: candidate.method.clone(),
            kernel: candidate.kernel.clone(),
        });
        components.push(ResolvedComponent {
            slot: format!("{}.method", operation.id),
            component: candidate.method.id.clone(),
            version: candidate.method.version.clone(),
            artifact_digest: candidate.method.artifact_digest.clone(),
        });
        components.push(ResolvedComponent {
            slot: format!("{}.kernel", operation.id),
            component: candidate.kernel.id.clone(),
            version: candidate.kernel.version.clone(),
            artifact_digest: candidate.kernel.artifact_digest.clone(),
        });
        if operation.operator == Operator::Conversion {
            conversions.push(NamedDigest {
                id: operation.id.clone(),
                digest: candidate.kernel.artifact_digest.clone(),
            });
        }
        evidence_references.extend_from_slice(candidate.evidence);
        let mut assurance_results = candidate.evidence.to_vec();
        assurance_results.sort();
        selected_bindings.push(CandidateEvidence {
            method_id: candidate.method.id.clone(),
            method_version: candidate.method.version.clone(),
            method_artifact_digest: candidate.method.artifact_digest.clone(),
            kernel_id: candidate.kernel.id.clone(),
            kernel_version: candidate.kernel.version.clone(),
            kernel_artifact_digest: candidate.kernel.artifact_digest.clone(),
            assurance_results,
        });
        tensor = operation.output.clone();
    }
    components.sort();
    conversions.sort();
    evidence_references.sort();
    evidence_references.dedup();
    selected_bindings.sort();
    selected_bindings.dedup();
    let selected_assurance_results: Vec<_> = evidence_references
        .iter()
        .map(|reference| {
            (*assurance_results
                .get(reference.id.as_str())
                .expect("selected evidence was validated"))
            .clone()
        })
        .collect();
    let region_program = RegionProgram {
        schema_version: REGION_PROGRAM_SCHEMA_VERSION.into(),
        logical_plan_digest: logical_digest.clone(),
        execution_role: request.context.execution_role.clone(),
        input: request.input.clone(),
        input_representation: request.input_representation,
        output: request.output.clone(),
        output_representation: request.output_representation,
        steps: region_steps,
    };
    let region_digest = region_program_digest(&region_program);
    let mut role_plans = request.context.static_role_plans.clone();
    role_plans.push(RolePlanReference {
        role: request.context.execution_role.clone(),
        digest: region_digest,
    });
    role_plans.sort();
    let execution = ExecutionPlan {
        schema_version: EXECUTION_PLAN_SCHEMA_VERSION.into(),
        logical_plan_digest: logical_digest.clone(),
        target: request.context.target.clone(),
        components,
        conversions,
        role_plans,
        material_requests: request.context.material_requests.clone(),
        resource_forecast: request.context.resource_forecast.clone(),
        evidence_references: evidence_references.clone(),
    };
    let execution_digest = execution_plan_digest(&execution);
    let mut locked_components: Vec<_> = execution
        .components
        .iter()
        .map(|component| VersionedArtifact {
            id: component.component.clone(),
            version: component.version.clone(),
            digest: component.artifact_digest.clone(),
        })
        .collect();
    locked_components.sort();
    locked_components.dedup();
    let lock = PlanLock {
        schema_version: PLAN_LOCK_SCHEMA_VERSION.into(),
        configuration_digest: request.configuration_digest.clone(),
        logical_plan_digest: logical_digest,
        execution_plan_digest: execution_digest,
        model_digest: request.context.model.digest.clone(),
        tokenizer_digest: request.context.tokenizer.digest.clone(),
        numeric_graph_digest: request.context.numeric_graph.digest.clone(),
        privacy_contract_digest: request.context.privacy_contract.digest.clone(),
        workload_digest: request.context.workload.digest.clone(),
        compiler: request.context.compiler.clone(),
        components: locked_components,
        evidence_references,
    };
    let compiled = CompiledPlan {
        logical,
        execution,
        lock,
        region_program,
        configuration_json: request.configuration_json.clone(),
        locked_context: request.context.clone(),
        privacy_contract: request.privacy_contract.clone(),
        assurance_results: selected_assurance_results,
        candidate_evidence: selected_bindings,
    };
    compiled.verify().map_err(|message| {
        vec![diagnostic(
            DiagnosticCode::InvalidContext,
            "compiled_plan",
            message,
        )]
    })?;
    Ok(compiled)
}

fn expected_components(region: &RegionProgram) -> Vec<ResolvedComponent> {
    let mut components = Vec::with_capacity(region.steps.len() * 2);
    for step in &region.steps {
        components.push(ResolvedComponent {
            slot: format!("{}.method", step.operation_id),
            component: step.method.id.clone(),
            version: step.method.version.clone(),
            artifact_digest: step.method.artifact_digest.clone(),
        });
        components.push(ResolvedComponent {
            slot: format!("{}.kernel", step.operation_id),
            component: step.kernel.id.clone(),
            version: step.kernel.version.clone(),
            artifact_digest: step.kernel.artifact_digest.clone(),
        });
    }
    components.sort();
    components
}

fn expected_conversions(region: &RegionProgram) -> Vec<NamedDigest> {
    let mut conversions: Vec<_> = region
        .steps
        .iter()
        .filter(|step| step.operator == Operator::Conversion)
        .map(|step| NamedDigest {
            id: step.operation_id.clone(),
            digest: step.kernel.artifact_digest.clone(),
        })
        .collect();
    conversions.sort();
    conversions
}

fn verify_region(region: &RegionProgram) -> Result<(), String> {
    if region.schema_version != REGION_PROGRAM_SCHEMA_VERSION
        || !valid_identity(&region.execution_role)
        || region.steps.is_empty()
    {
        return Err("invalid region program header".into());
    }
    let mut tensor = &region.input;
    let mut representation = region.input_representation;
    for step in &region.steps {
        if !valid_identity(&step.operation_id)
            || !valid_identity(&step.method.id)
            || !valid_identity(&step.method.version)
            || !valid_identity(&step.kernel.id)
            || !valid_identity(&step.kernel.version)
            || !valid_identity(&step.kernel.method_id)
            || !valid_identity(&step.kernel.method_version)
            || &step.input != tensor
            || step.input_representation != representation
            || step.method.operator != step.operator
            || step.method.input_representation != step.input_representation
            || step.method.output_representation != step.output_representation
            || step.kernel.method_id != step.method.id
            || step.kernel.method_version != step.method.version
            || step.kernel.input_numeric != step.input.numeric
            || step.kernel.output_numeric != step.output.numeric
            || !implementation_matches(step.operator, step.kernel.implementation)
        {
            return Err("region steps are incomplete or inconsistent".into());
        }
        validate_tensor(
            &LogicalOperation {
                id: step.operation_id.clone(),
                operator: step.operator,
                output: step.output.clone(),
                input_representation: step.input_representation,
                output_representation: step.output_representation,
            },
            &step.input,
        )
        .map_err(|_| "region tensor semantics are inconsistent")?;
        tensor = &step.output;
        representation = step.output_representation;
    }
    if tensor != &region.output || representation != region.output_representation {
        return Err("region output does not match final step".into());
    }
    Ok(())
}

fn verify_compiled_plan(compiled: &CompiledPlan) -> Result<(), String> {
    let logical = &compiled.logical;
    let execution = &compiled.execution;
    let lock = &compiled.lock;
    let region = &compiled.region_program;
    let configuration =
        validate_configuration_json(&compiled.configuration_json, &logical.configuration_digest)?;
    if configuration_profile(&configuration) != Some(logical.profile.as_str()) {
        return Err("configuration profile does not match canonical plan".into());
    }
    if logical.schema_version != LOGICAL_PLAN_SCHEMA_VERSION
        || execution.schema_version != EXECUTION_PLAN_SCHEMA_VERSION
        || lock.schema_version != PLAN_LOCK_SCHEMA_VERSION
    {
        return Err("unsupported canonical plan schema version".into());
    }
    if logical.roles.is_empty()
        || !sorted_unique(&logical.roles)
        || !sorted_unique(&logical.required_claim_ids)
        || !valid_identity(&logical.profile)
        || !valid_named(&logical.model)
        || !valid_named(&logical.tokenizer)
        || !valid_named(&logical.semantic_graph)
        || !valid_named(&logical.numeric_graph)
        || !valid_named(&logical.protected_graph)
        || !valid_named(&logical.privacy_contract)
        || !valid_named(&logical.workload)
    {
        return Err("logical plan contains malformed or unordered identities".into());
    }
    if lock.configuration_digest != logical.configuration_digest {
        return Err("configuration digest differs between logical plan and lock".into());
    }
    let context = &compiled.locked_context;
    if context.schema_version != LOCKED_CONTEXT_SCHEMA_VERSION
        || logical.profile != context.profile
        || logical.model != context.model
        || logical.tokenizer != context.tokenizer
        || logical.semantic_graph != context.semantic_graph
        || logical.numeric_graph != context.numeric_graph
        || logical.protected_graph != context.protected_graph
        || logical.roles != context.roles
        || logical.privacy_contract != context.privacy_contract
        || logical.workload != context.workload
        || execution.target != context.target
        || execution.material_requests != context.material_requests
        || execution.resource_forecast != context.resource_forecast
        || lock.compiler != context.compiler
        || region.execution_role != context.execution_role
    {
        return Err("canonical plans differ from locked context sidecar".into());
    }
    let expected_static_roles: Vec<_> = context
        .roles
        .iter()
        .filter(|role| role.as_str() != context.execution_role)
        .map(String::as_str)
        .collect();
    let actual_static_roles: Vec<_> = context
        .static_role_plans
        .iter()
        .map(|plan| plan.role.as_str())
        .collect();
    if context.roles.is_empty()
        || !sorted_unique(&context.roles)
        || !context.roles.contains(&context.execution_role)
        || !sorted_unique(&context.static_role_plans)
        || expected_static_roles != actual_static_roles
    {
        return Err("locked context role plans are incomplete or unordered".into());
    }
    let logical_digest = logical_plan_digest(logical);
    if execution.logical_plan_digest != logical_digest || lock.logical_plan_digest != logical_digest
    {
        return Err("logical plan digest does not match execution plan and lock".into());
    }
    verify_region(region)?;
    if region
        .steps
        .iter()
        .any(|step| step.operator == Operator::Silu)
    {
        let [step] = region.steps.as_slice() else {
            return Err("the Q7 SiLU executor supports only a singleton region".into());
        };
        if logical.profile != SILU_Q7_EXPERIMENT_PROFILE {
            return Err(format!(
                "Q7 SiLU requires profile {SILU_Q7_EXPERIMENT_PROFILE}"
            ));
        }
        if logical.numeric_graph.id != SILU_Q7_NUMERIC_GRAPH_ID
            || logical.protected_graph.id != SILU_Q7_PROTECTED_GRAPH_ID
        {
            return Err("Q7 SiLU plan uses the wrong installed graph identity".into());
        }
        if context.compiler.id != SILU_Q7_COMPILER_ID
            || context.compiler.version != env!("CARGO_PKG_VERSION")
            || context.compiler.digest != silu_q7_compiler_artifact_digest()
        {
            return Err("compiled Q7 SiLU compiler artifact is invalid".into());
        }
        if let Some(diagnostic) = validate_silu_q7_descriptor(&step.method, &step.kernel) {
            return Err(diagnostic.message);
        }
    }
    if region.logical_plan_digest != logical_digest {
        return Err("region program references wrong logical plan".into());
    }
    let mut expected_role_plans = context.static_role_plans.clone();
    expected_role_plans.push(RolePlanReference {
        role: region.execution_role.clone(),
        digest: region_program_digest(region),
    });
    expected_role_plans.sort();
    if execution.role_plans != expected_role_plans {
        return Err("role plans do not cover roles or bind region program digest".into());
    }
    let operations: Vec<_> = region
        .steps
        .iter()
        .map(|step| LogicalOperation {
            id: step.operation_id.clone(),
            operator: step.operator,
            output: step.output.clone(),
            input_representation: step.input_representation,
            output_representation: step.output_representation,
        })
        .collect();
    let graph_digests = region_graph_digests(
        &region.input,
        region.input_representation,
        &region.output,
        region.output_representation,
        &operations,
    );
    if logical.semantic_graph.digest != graph_digests.semantic_graph
        || logical.numeric_graph.digest != graph_digests.numeric_graph
        || logical.protected_graph.digest != graph_digests.protected_graph
    {
        return Err("region intent does not match locked graph digests".into());
    }
    if execution.components != expected_components(region)
        || execution.conversions != expected_conversions(region)
        || !sorted_unique(&execution.components)
        || !sorted_unique(&execution.conversions)
        || !sorted_unique(&execution.material_requests)
        || !sorted_unique(&execution.evidence_references)
        || execution.components.iter().any(|component| {
            !valid_identity(&component.slot)
                || !valid_identity(&component.component)
                || !valid_identity(&component.version)
        })
        || execution.conversions.iter().any(|item| !valid_named(item))
        || execution
            .material_requests
            .iter()
            .any(|item| !valid_named(item))
        || execution
            .evidence_references
            .iter()
            .any(|item| !valid_identity(&item.id))
        || !valid_named(&execution.target)
        || !valid_named(&execution.resource_forecast)
    {
        return Err("execution manifest does not match region program".into());
    }
    let execution_digest = execution_plan_digest(execution);
    if lock.execution_plan_digest != execution_digest {
        return Err("execution plan digest does not match lock".into());
    }
    if lock.configuration_digest != logical.configuration_digest
        || lock.model_digest != logical.model.digest
        || lock.tokenizer_digest != logical.tokenizer.digest
        || lock.numeric_graph_digest != logical.numeric_graph.digest
        || lock.privacy_contract_digest != logical.privacy_contract.digest
        || lock.workload_digest != logical.workload.digest
        || lock.evidence_references != execution.evidence_references
        || !sorted_unique(&lock.components)
        || !sorted_unique(&lock.evidence_references)
        || !valid_identity(&lock.compiler.id)
        || !valid_identity(&lock.compiler.version)
        || lock
            .components
            .iter()
            .any(|component| !valid_identity(&component.id) || !valid_identity(&component.version))
        || lock
            .evidence_references
            .iter()
            .any(|item| !valid_identity(&item.id))
    {
        return Err("plan lock does not commit canonical plan inputs".into());
    }
    let mut components: Vec<_> = execution
        .components
        .iter()
        .map(|component| VersionedArtifact {
            id: component.component.clone(),
            version: component.version.clone(),
            digest: component.artifact_digest.clone(),
        })
        .collect();
    components.sort();
    components.dedup();
    if lock.components != components {
        return Err("plan lock component artifacts differ from execution plan".into());
    }

    let privacy = &compiled.privacy_contract;
    if privacy.schema_version != PRIVACY_CONTRACT_SCHEMA_VERSION
        || privacy.id != logical.privacy_contract.id
        || privacy_contract_digest(privacy) != logical.privacy_contract.digest
        || privacy_contract_digest(privacy) != lock.privacy_contract_digest
        || privacy.online_parties == 0
        || !privacy
            .required_claims
            .windows(2)
            .all(|pair| pair[0].claim_id < pair[1].claim_id)
        || privacy.required_claims.iter().any(|requirement| {
            !valid_identity(&requirement.claim_id)
                || requirement.accepted_outcomes.is_empty()
                || requirement.accepted_implementation_refinements.is_empty()
        })
    {
        return Err("privacy contract sidecar does not match locked contract".into());
    }
    let claim_ids: Vec<_> = privacy
        .required_claims
        .iter()
        .map(|requirement| requirement.claim_id.clone())
        .collect();
    if logical.required_claim_ids != claim_ids {
        return Err("logical required claims differ from privacy contract".into());
    }
    if !compiled
        .assurance_results
        .windows(2)
        .all(|pair| pair[0].id < pair[1].id)
        || compiled
            .assurance_results
            .iter()
            .any(|result| !validate_assurance_result(result))
        || !sorted_unique(&compiled.candidate_evidence)
    {
        return Err("assurance sidecar is malformed or unordered".into());
    }
    let assurance_results: BTreeMap<_, _> = compiled
        .assurance_results
        .iter()
        .map(|result| (result.id.as_str(), result))
        .collect();
    let actual_evidence_references: Vec<_> = compiled
        .assurance_results
        .iter()
        .map(|result| EvidenceReference {
            id: result.id.clone(),
            digest: assurance_result_digest(result),
        })
        .collect();
    if actual_evidence_references != execution.evidence_references {
        return Err("assurance result records do not match execution references".into());
    }
    let candidate_evidence: BTreeMap<_, _> = compiled
        .candidate_evidence
        .iter()
        .map(|binding| {
            (
                (
                    binding.method_id.as_str(),
                    binding.method_version.as_str(),
                    &binding.method_artifact_digest,
                    binding.kernel_id.as_str(),
                    binding.kernel_version.as_str(),
                    &binding.kernel_artifact_digest,
                ),
                binding,
            )
        })
        .collect();
    let expected_candidate_keys: BTreeSet<_> = region
        .steps
        .iter()
        .map(|step| {
            (
                step.method.id.as_str(),
                step.method.version.as_str(),
                &step.method.artifact_digest,
                step.kernel.id.as_str(),
                step.kernel.version.as_str(),
                &step.kernel.artifact_digest,
            )
        })
        .collect();
    if candidate_evidence.keys().copied().collect::<BTreeSet<_>>() != expected_candidate_keys {
        return Err("candidate evidence sidecar does not match selected region candidates".into());
    }
    let mut referenced_evidence = Vec::new();
    for step in &region.steps {
        let key = (
            step.method.id.as_str(),
            step.method.version.as_str(),
            &step.method.artifact_digest,
            step.kernel.id.as_str(),
            step.kernel.version.as_str(),
            &step.kernel.artifact_digest,
        );
        let binding = candidate_evidence
            .get(&key)
            .ok_or("selected candidate has no evidence binding")?;
        if !sorted_unique(&binding.assurance_results) {
            return Err("candidate evidence references are unordered or duplicated".into());
        }
        for reference in &binding.assurance_results {
            let result = assurance_results
                .get(reference.id.as_str())
                .ok_or("candidate references missing assurance result")?;
            if assurance_result_digest(result) != reference.digest {
                return Err("candidate assurance reference digest is invalid".into());
            }
        }
        let candidate_id = format!("{}@{}/{}@{}", key.0, key.1, key.3, key.4);
        if !validate_method_properties(privacy, &step.method).is_empty()
            || !validate_candidate_assurance(
                privacy,
                &candidate_id,
                [&step.method.artifact_digest, &step.kernel.artifact_digest],
                &binding.assurance_results,
                &assurance_results,
            )
            .is_empty()
        {
            return Err("selected candidate no longer satisfies privacy contract".into());
        }
        referenced_evidence.extend(binding.assurance_results.iter().cloned());
    }
    referenced_evidence.sort();
    referenced_evidence.dedup();
    if referenced_evidence != execution.evidence_references {
        return Err("selected candidate evidence differs from execution references".into());
    }
    Ok(())
}

pub fn region_program_bytes(program: &RegionProgram) -> Vec<u8> {
    canonical_bytes(program)
}

pub fn region_program_digest(program: &RegionProgram) -> Digest {
    pllm_types::canonical_digest(REGION_PROGRAM_SCHEMA_VERSION, program)
}

/// Extract one semantic linear operation into the compiler's wrap32 matrix representation.
pub fn lower_model_linear_operation(
    plan: &DecoderPlan,
    mode: DecoderMode,
    operation_id: &str,
) -> Result<(TensorType, LogicalOperation), String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    let operation = graph
        .operations
        .iter()
        .find(|operation| operation.id == operation_id)
        .ok_or_else(|| format!("semantic operation {operation_id} is absent from {mode:?}"))?;
    let region = lower_model_linear_region(graph, mode, operation)?;
    Ok((region.input, region.operation))
}

/// Extract every semantic linear operation with the weight identity needed by execution.
pub fn lower_model_linear_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelLinearRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::Linear)
        .map(|operation| lower_model_linear_region(graph, mode, operation))
        .collect()
}

/// Extract supported semantic reshape operations into exact layout transformations.
pub fn lower_model_reshape_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelReshapeRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::Reshape)
        .map(|operation| lower_model_reshape_region(graph, mode, operation))
        .collect()
}

/// Execute the head/sequence permutation encoded by a semantic reshape layout.
pub fn execute_model_reshape(
    region: &ModelReshapeRegion,
    input: &[u32],
) -> Result<Vec<u32>, String> {
    let input_elements = tensor_elements(&region.input.shape)?;
    let output_elements = tensor_elements(&region.output.shape)?;
    if input_elements != output_elements {
        return Err("semantic reshape changes the element count".into());
    }
    if input.len() != input_elements {
        return Err(format!(
            "semantic reshape requires {input_elements} input elements, received {}",
            input.len()
        ));
    }
    let mut output = input.to_vec();
    match region.layout {
        ModelReshapeLayout::BatchHeadsSequenceFeature => {
            let [batch, sequence, hidden] = region.input.shape.as_slice() else {
                return Err("batch_heads_sequence_feature requires rank-3 input".into());
            };
            let [output_batch, heads, output_sequence, feature] = region.output.shape.as_slice()
            else {
                return Err("batch_heads_sequence_feature requires rank-4 output".into());
            };
            if batch != output_batch
                || sequence != output_sequence
                || hidden
                    != &heads
                        .checked_mul(*feature)
                        .ok_or("reshape shape overflows u64")?
            {
                return Err("batch_heads_sequence_feature shapes are incompatible".into());
            }
            for b in 0..*batch {
                for s in 0..*sequence {
                    for h in 0..*heads {
                        for d in 0..*feature {
                            let source = (((b * sequence + s) * heads + h) * feature + d) as usize;
                            let target = (((b * heads + h) * sequence + s) * feature + d) as usize;
                            output[target] = input[source];
                        }
                    }
                }
            }
        }
        ModelReshapeLayout::BatchSequenceHidden => {
            let [batch, heads, sequence, feature] = region.input.shape.as_slice() else {
                return Err("batch_sequence_hidden requires rank-4 input".into());
            };
            let [output_batch, output_sequence, hidden] = region.output.shape.as_slice() else {
                return Err("batch_sequence_hidden requires rank-3 output".into());
            };
            if batch != output_batch
                || sequence != output_sequence
                || hidden
                    != &heads
                        .checked_mul(*feature)
                        .ok_or("reshape shape overflows u64")?
            {
                return Err("batch_sequence_hidden shapes are incompatible".into());
            }
            for b in 0..*batch {
                for h in 0..*heads {
                    for s in 0..*sequence {
                        for d in 0..*feature {
                            let source = (((b * heads + h) * sequence + s) * feature + d) as usize;
                            let target = (((b * sequence + s) * heads + h) * feature + d) as usize;
                            output[target] = input[source];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Extract semantic residual additions with both graph input identities intact.
pub fn lower_model_residual_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelResidualRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::ResidualAdd)
        .map(|operation| lower_model_residual_region(graph, mode, operation))
        .collect()
}

/// Execute a semantic residual addition in the wrap32 ring.
pub fn execute_model_residual(
    region: &ModelResidualRegion,
    left: &[u32],
    right: &[u32],
) -> Result<Vec<u32>, String> {
    if region.input.numeric != NumericType::Wrap32
        || region.output.numeric != NumericType::Wrap32
        || region.input.shape != region.output.shape
    {
        return Err("semantic residual requires equal wrap32 input and output tensors".into());
    }
    let elements = tensor_elements(&region.input.shape)?;
    if left.len() != elements || right.len() != elements {
        return Err(format!(
            "semantic residual requires {elements} elements per input, received {} and {}",
            left.len(),
            right.len()
        ));
    }
    pllm_core::add_wrap32(left, right)
}

/// Extract semantic output heads with their exact checkpoint weight identity.
pub fn lower_model_output_head_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelOutputHeadRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::OutputHead)
        .map(|operation| lower_model_output_head_region(graph, mode, operation))
        .collect()
}

/// Execute one semantic output head through the wrap32 matrix kernel.
pub fn execute_model_output_head(
    region: &ModelOutputHeadRegion,
    weights: &[u8],
    input: &[u32],
    threads: usize,
    simd: bool,
) -> Result<Vec<u32>, String> {
    if region.input.numeric != NumericType::Wrap32
        || region.output.numeric != NumericType::Wrap32
        || region.input.shape.len() < 2
        || region.output.shape.len() != region.input.shape.len()
        || region.input.shape[..region.input.shape.len() - 1]
            != region.output.shape[..region.output.shape.len() - 1]
    {
        return Err(
            "semantic output head requires matching wrap32 leading dimensions and one feature dimension"
                .into(),
        );
    }
    let leading = &region.input.shape[..region.input.shape.len() - 1];
    let batch = leading.iter().try_fold(1_u64, |rows, dimension| {
        rows.checked_mul(*dimension)
            .ok_or("semantic output-head row count overflows u64")
    })?;
    let batch = usize::try_from(batch).map_err(|_| "output-head row count exceeds usize")?;
    let columns = usize::try_from(*region.input.shape.last().unwrap())
        .map_err(|_| "output-head input width exceeds usize")?;
    let rows = usize::try_from(*region.output.shape.last().unwrap())
        .map_err(|_| "output-head output width exceeds usize")?;
    let matrix = pllm_core::Matrix::new(weights, rows, columns)?;
    let executor = pllm_core::Executor::new(threads, simd)?;
    matrix.wrap32(&executor, input, batch)
}

/// Extract physical-last and last-valid token selections.
pub fn lower_model_last_token_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelLastTokenRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::LastToken)
        .map(|operation| lower_model_last_token_region(graph, mode, operation))
        .collect()
}

/// Select the last token on the semantic sequence axis — physically or by
/// valid length depending on the region's selection mode.
pub fn execute_model_last_token(
    region: &ModelLastTokenRegion,
    input: &[u32],
    sequence_lengths: Option<&[u64]>,
) -> Result<Vec<u32>, String> {
    if region.input.numeric != NumericType::Wrap32
        || region.output.numeric != NumericType::Wrap32
        || region.input.shape.len() < 2
        || region.axis >= region.input.shape.len()
        || region.input.shape.contains(&0)
    {
        return Err("semantic last-token region has an invalid wrap32 input contract".into());
    }
    let mut expected_output_shape = region.input.shape.clone();
    expected_output_shape.remove(region.axis);
    if region.output.shape != expected_output_shape {
        return Err("semantic last-token output shape does not remove its selected axis".into());
    }
    let input_elements = tensor_elements(&region.input.shape)?;
    if input.len() != input_elements {
        return Err("last-token input length does not match its semantic shape".into());
    }
    match region.selection {
        ModelLastTokenSelection::PhysicalLast => {
            if sequence_lengths.is_some() || region.sequence_lengths_input_id.is_some() {
                return Err("physical-last selection does not accept sequence lengths".into());
            }
            let outer = tensor_elements(&region.input.shape[..region.axis])?;
            let axis = usize::try_from(region.input.shape[region.axis])
                .map_err(|_| "last-token axis extent exceeds usize")?;
            let inner = tensor_elements(&region.input.shape[region.axis + 1..])?;
            let output_elements = outer
                .checked_mul(inner)
                .ok_or("last-token output element count overflows usize")?;
            let mut output = Vec::with_capacity(output_elements);
            for outer_index in 0..outer {
                let start = outer_index
                    .checked_mul(axis)
                    .and_then(|offset| offset.checked_add(axis - 1))
                    .and_then(|offset| offset.checked_mul(inner))
                    .ok_or("last-token source offset overflows usize")?;
                output.extend_from_slice(&input[start..start + inner]);
            }
            Ok(output)
        }
        ModelLastTokenSelection::LastValid => {
            if region.sequence_lengths_input_id.as_deref() != Some("input.sequence_lengths")
                || region.axis != 1
                || region.input.shape.len() < 2
            {
                return Err(
                    "last-valid selection requires the input.sequence_lengths input on axis 1"
                        .into(),
                );
            }
            let lengths = sequence_lengths
                .ok_or("last-valid selection requires the sequence lengths input")?;
            let batch = usize::try_from(region.input.shape[0])
                .map_err(|_| "last-token batch exceeds usize")?;
            if lengths.len() != batch {
                return Err(
                    "last-token sequence lengths must provide exactly one length per batch".into(),
                );
            }
            let axis = usize::try_from(region.input.shape[1])
                .map_err(|_| "last-token axis extent exceeds usize")?;
            if lengths
                .iter()
                .any(|length| *length < 1 || *length > region.input.shape[1])
            {
                return Err(
                    "last-token sequence lengths must be within 1 and the sequence extent".into(),
                );
            }
            let inner = tensor_elements(&region.input.shape[2..])?;
            let mut output = Vec::with_capacity(
                batch
                    .checked_mul(inner)
                    .ok_or("last-token output element count overflows usize")?,
            );
            for (batch_index, &length) in lengths.iter().enumerate() {
                let row = usize::try_from(length - 1)
                    .map_err(|_| "last-token sequence length exceeds usize")?;
                let start = batch_index
                    .checked_mul(axis)
                    .and_then(|offset| offset.checked_add(row))
                    .and_then(|offset| offset.checked_mul(inner))
                    .ok_or("last-token source offset overflows usize")?;
                output.extend_from_slice(&input[start..start + inner]);
            }
            Ok(output)
        }
    }
}

pub fn lower_model_greedy_token_selection_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelGreedyTokenSelectionRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::GreedyTokenSelection)
        .map(|operation| {
            let operation_id = operation.id.as_str();
            let [input_id] = operation.inputs.as_slice() else {
                return Err(format!(
                    "semantic greedy selection operation {operation_id} must have exactly one input"
                ));
            };
            let input = graph
                .operations
                .iter()
                .find(|candidate| candidate.id == *input_id)
                .ok_or_else(|| {
                    format!(
                        "semantic greedy selection operation {operation_id} references missing input {input_id}"
                    )
                })?;
            if input.operator != ModelOperator::OutputHead {
                return Err(format!(
                    "semantic greedy selection operation {operation_id} must consume an output head"
                ));
            }
            if operation.attributes
                != serde_json::json!({"policy": "pllm.greedy.v1", "source": "execution_policy"})
            {
                return Err(format!(
                    "semantic greedy selection operation {operation_id} does not carry the pllm.greedy.v1 execution policy"
                ));
            }
            if input.output_shape.len() != 2
                || input.output_shape.contains(&0)
                || operation.output_shape != [input.output_shape[0]]
            {
                return Err(format!(
                    "semantic greedy selection operation {operation_id} requires a rank-two [batch, vocabulary] input and a [batch] output"
                ));
            }
            tensor_elements(&input.output_shape)?;
            tensor_elements(&operation.output_shape)?;
            Ok(ModelGreedyTokenSelectionRegion {
                mode,
                layer: operation.layer,
                operation_id: operation.id.clone(),
                input_id: input_id.clone(),
                input: TensorType {
                    numeric: NumericType::Wrap32,
                    shape: input.output_shape.clone(),
                },
                output: TensorType {
                    numeric: NumericType::TokenIdU32,
                    shape: operation.output_shape.clone(),
                },
            })
        })
        .collect()
}

pub fn execute_model_greedy_token_selection(
    region: &ModelGreedyTokenSelectionRegion,
    logits: &[u32],
) -> Result<Vec<u32>, String> {
    if region.input.numeric != NumericType::Wrap32
        || region.output.numeric != NumericType::TokenIdU32
        || region.input.shape.len() != 2
        || region.input.shape.contains(&0)
        || region.output.shape != [region.input.shape[0]]
    {
        return Err(
            "semantic greedy selection requires a wrap32 [batch, vocabulary] input and a token-id [batch] output"
                .into(),
        );
    }
    let batch = usize::try_from(region.input.shape[0])
        .map_err(|_| "greedy selection batch exceeds usize")?;
    let vocabulary = usize::try_from(region.input.shape[1])
        .map_err(|_| "greedy selection vocabulary exceeds usize")?;
    let elements = batch
        .checked_mul(vocabulary)
        .ok_or("greedy selection element count overflows usize")?;
    if logits.len() != elements {
        return Err("greedy selection logits length does not match its semantic shape".into());
    }
    let mut output = Vec::with_capacity(batch);
    for row in logits.chunks_exact(vocabulary) {
        let mut best_index = 0_usize;
        let mut best_value = row[0] as i32;
        for (index, &value) in row.iter().enumerate().skip(1) {
            let value = value as i32;
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }
        output.push(
            u32::try_from(best_index)
                .map_err(|_| "greedy selection vocabulary index exceeds u32")?,
        );
    }
    Ok(output)
}

pub fn lower_model_token_feedback_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelTokenFeedbackRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::TokenFeedback)
        .map(|operation| {
            let operation_id = operation.id.as_str();
            let [input_id] = operation.inputs.as_slice() else {
                return Err(format!(
                    "semantic token-feedback operation {operation_id} must have exactly one input"
                ));
            };
            let input = graph
                .operations
                .iter()
                .find(|candidate| candidate.id == *input_id)
                .ok_or_else(|| {
                    format!(
                        "semantic token-feedback operation {operation_id} references missing input {input_id}"
                    )
                })?;
            if operation.operator != ModelOperator::TokenFeedback
                || input.operator != ModelOperator::GreedyTokenSelection
            {
                return Err(format!(
                    "semantic token-feedback operation {operation_id} must consume a greedy token selection"
                ));
            }
            if operation.attributes
                != serde_json::json!({"policy": "pllm.greedy.v1", "source": "execution_policy"})
            {
                return Err(format!(
                    "semantic token-feedback operation {operation_id} does not carry the pllm.greedy.v1 execution policy"
                ));
            }
            if input.output_shape != [graph.batch]
                || operation.output_shape != [graph.batch, 1]
            {
                return Err(format!(
                    "semantic token-feedback operation {operation_id} requires a [batch] selection input and a [batch, 1] output"
                ));
            }
            tensor_elements(&input.output_shape)?;
            tensor_elements(&operation.output_shape)?;
            Ok(ModelTokenFeedbackRegion {
                mode,
                layer: operation.layer,
                operation_id: operation.id.clone(),
                input_id: input_id.clone(),
                input: TensorType {
                    numeric: NumericType::TokenIdU32,
                    shape: input.output_shape.clone(),
                },
                output: TensorType {
                    numeric: NumericType::TokenIdU32,
                    shape: operation.output_shape.clone(),
                },
            })
        })
        .collect()
}

pub fn execute_model_token_feedback(
    region: &ModelTokenFeedbackRegion,
    token_ids: &[u32],
) -> Result<Vec<u32>, String> {
    if region.input.numeric != NumericType::TokenIdU32
        || region.output.numeric != NumericType::TokenIdU32
        || region.input.shape.len() != 1
        || region.output.shape != [region.input.shape[0], 1]
        || region.input.shape.contains(&0)
    {
        return Err(
            "semantic token feedback requires a token-id [batch] input and a token-id [batch, 1] output"
                .into(),
        );
    }
    let batch =
        usize::try_from(region.input.shape[0]).map_err(|_| "token-feedback batch exceeds usize")?;
    if token_ids.len() != batch {
        return Err("token-feedback input length does not match its semantic shape".into());
    }
    Ok(token_ids.to_vec())
}

pub fn lower_model_softmax_q30_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelSoftmaxQ30Region>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::Softmax)
        .map(|operation| {
            let operation_id = operation.id.as_str();
            let [input_id] = operation.inputs.as_slice() else {
                return Err(format!(
                    "semantic softmax operation {operation_id} must have exactly one input"
                ));
            };
            let input = graph
                .operations
                .iter()
                .find(|candidate| candidate.id == *input_id)
                .ok_or_else(|| {
                    format!(
                        "semantic softmax operation {operation_id} references missing input {input_id}"
                    )
                })?;
            if input.operator != ModelOperator::CausalMask {
                return Err(format!(
                    "semantic softmax operation {operation_id} must consume a causal mask"
                ));
            }
            if operation.output_shape.len() != 4
                || operation.output_shape != input.output_shape
                || operation.output_shape.contains(&0)
            {
                return Err(format!(
                    "semantic softmax operation {operation_id} requires a nonzero rank-four shape matching its mask input"
                ));
            }
            let axis = operation
                .attributes
                .get("axis")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| {
                    format!("semantic softmax operation {operation_id} has invalid axis")
                })?;
            let normalized = if axis < 0 { axis + 4 } else { axis };
            if normalized != 3 {
                return Err(format!(
                    "semantic softmax operation {operation_id} must normalize over the final axis"
                ));
            }
            tensor_elements(&input.output_shape)?;
            tensor_elements(&operation.output_shape)?;
            Ok(ModelSoftmaxQ30Region {
                mode,
                layer: operation.layer,
                operation_id: operation.id.clone(),
                input_id: input_id.clone(),
                axis: 3,
                numeric_profile: pllm_core::SOFTMAX_Q20_TO_Q30_PROFILE.to_owned(),
                input: TensorType {
                    numeric: NumericType::SignedFixedQ20,
                    shape: input.output_shape.clone(),
                },
                output: TensorType {
                    numeric: NumericType::UnsignedFixedQ30,
                    shape: operation.output_shape.clone(),
                },
            })
        })
        .collect()
}

pub fn execute_model_softmax_q30(
    region: &ModelSoftmaxQ30Region,
    scores: &[i64],
    allowed: &[bool],
    policy: pllm_core::SoftmaxQ30Policy,
) -> Result<pllm_core::SoftmaxProbabilitiesQ30, String> {
    if region.numeric_profile != pllm_core::SOFTMAX_Q20_TO_Q30_PROFILE
        || region.input.numeric != NumericType::SignedFixedQ20
        || region.output.numeric != NumericType::UnsignedFixedQ30
        || region.axis != 3
        || region.input.shape.len() != 4
        || region.input.shape != region.output.shape
        || region.input.shape.contains(&0)
    {
        return Err(
            "semantic softmax requires the q20-to-q30 profile, matching nonzero rank-four shapes, and axis three"
                .into(),
        );
    }
    let shape: [usize; 4] = region
        .input
        .shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| "softmax dimension exceeds usize".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "softmax shape must have rank four".to_owned())?;
    pllm_core::softmax_q20_to_q30(scores, allowed, shape, policy).map_err(|error| error.to_string())
}

pub fn lower_model_attention_values_q10_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelAttentionValuesQ10Region>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::AttentionValues)
        .map(|operation| {
            let operation_id = operation.id.as_str();
            let [probabilities_id, values_id] = operation.inputs.as_slice() else {
                return Err(format!(
                    "semantic attention-values operation {operation_id} must have exactly two inputs"
                ));
            };
            let producer = |id: &str| {
                graph
                    .operations
                    .iter()
                    .find(|candidate| candidate.id == id)
                    .ok_or_else(|| {
                        format!(
                            "semantic attention-values operation {operation_id} references missing input {id}"
                        )
                    })
            };
            let probabilities = producer(probabilities_id)?;
            if probabilities.operator != ModelOperator::Softmax {
                return Err(format!(
                    "semantic attention-values operation {operation_id} must consume a softmax"
                ));
            }
            let values = producer(values_id)?;
            let layout = match values.operator {
                ModelOperator::CacheSuffix => {
                    if values.state_kind != Some(StateKind::Value) {
                        return Err(format!(
                            "semantic attention-values operation {operation_id} must consume the value cache view"
                        ));
                    }
                    pllm_core::AttentionValueQ10Layout::GroupedQuery
                }
                ModelOperator::SecureGather => {
                    if values.state_kind != Some(StateKind::Value)
                        || values
                            .attributes
                            .get("output_semantics")
                            .and_then(serde_json::Value::as_str)
                            != Some("per_query_head_cache_window")
                    {
                        return Err(format!(
                            "semantic attention-values operation {operation_id} must consume a per-query-head value gather"
                        ));
                    }
                    pllm_core::AttentionValueQ10Layout::PerQueryHeadWindow
                }
                _ => {
                    return Err(format!(
                        "semantic attention-values operation {operation_id} must consume a cache view or value gather"
                    ))
                }
            };
            if probabilities.output_shape.len() != 4
                || probabilities.output_shape.contains(&0)
                || operation.output_shape.len() != 4
                || operation.output_shape.contains(&0)
            {
                return Err(format!(
                    "semantic attention-values operation {operation_id} requires nonzero rank-four probability and output shapes"
                ));
            }
            let group_size = operation
                .attributes
                .get("group_size")
                .and_then(serde_json::Value::as_u64)
                .filter(|group_size| *group_size > 0)
                .ok_or_else(|| {
                    format!(
                        "semantic attention-values operation {operation_id} has invalid group_size"
                    )
                })?;
            let (batch, query_heads, queries, keys) = (
                probabilities.output_shape[0],
                probabilities.output_shape[1],
                probabilities.output_shape[2],
                probabilities.output_shape[3],
            );
            match layout {
                pllm_core::AttentionValueQ10Layout::GroupedQuery => {
                    if values.output_shape.len() != 4
                        || values.output_shape.contains(&0)
                        || values.output_shape[0] != batch
                        || values.output_shape[2] != keys
                        || query_heads % values.output_shape[1] != 0
                        || group_size != query_heads / values.output_shape[1]
                    {
                        return Err(format!(
                            "semantic attention-values operation {operation_id} requires a [batch, kv_heads, keys, depth] grouped-query value view"
                        ));
                    }
                    if operation.output_shape
                        != [batch, query_heads, queries, values.output_shape[3]]
                    {
                        return Err(format!(
                            "semantic attention-values operation {operation_id} output shape does not match the grouped-query contraction"
                        ));
                    }
                }
                pllm_core::AttentionValueQ10Layout::PerQueryHeadWindow => {
                    if values.output_shape.len() != 5
                        || values.output_shape.contains(&0)
                        || values.output_shape[..4]
                            != [batch, query_heads, queries, keys]
                    {
                        return Err(format!(
                            "semantic attention-values operation {operation_id} requires a [batch, heads, query, keys, depth] per-query-head value window"
                        ));
                    }
                    if operation.output_shape
                        != [batch, query_heads, queries, values.output_shape[4]]
                    {
                        return Err(format!(
                            "semantic attention-values operation {operation_id} output shape does not match the per-query-head contraction"
                        ));
                    }
                }
            }
            tensor_elements(&probabilities.output_shape)?;
            tensor_elements(&values.output_shape)?;
            tensor_elements(&operation.output_shape)?;
            Ok(ModelAttentionValuesQ10Region {
                mode,
                layer: operation.layer,
                operation_id: operation.id.clone(),
                probabilities_input_id: probabilities_id.clone(),
                values_input_id: values_id.clone(),
                numeric_profile: pllm_core::ATTENTION_VALUE_Q30_Q10_PROFILE.to_owned(),
                layout,
                probabilities: TensorType {
                    numeric: NumericType::UnsignedFixedQ30,
                    shape: probabilities.output_shape.clone(),
                },
                values: TensorType {
                    numeric: NumericType::SignedFixedQ10,
                    shape: values.output_shape.clone(),
                },
                output: TensorType {
                    numeric: NumericType::SignedFixedQ10,
                    shape: operation.output_shape.clone(),
                },
            })
        })
        .collect()
}

pub fn execute_model_attention_values_q10(
    region: &ModelAttentionValuesQ10Region,
    probabilities: &pllm_core::SoftmaxProbabilitiesQ30,
    values: &[i16],
    policy: pllm_core::AttentionValueQ10Policy,
) -> Result<pllm_core::AttentionValuesQ10, String> {
    if region.numeric_profile != pllm_core::ATTENTION_VALUE_Q30_Q10_PROFILE
        || region.probabilities.numeric != NumericType::UnsignedFixedQ30
        || region.values.numeric != NumericType::SignedFixedQ10
        || region.output.numeric != NumericType::SignedFixedQ10
        || region.probabilities.shape.len() != 4
        || region.probabilities.shape.contains(&0)
        || region.output.shape.len() != 4
        || region.output.shape.contains(&0)
        || region.values.shape.contains(&0)
    {
        return Err(
            "semantic attention-values requires the q30/q10 profile and nonzero rank-four shapes"
                .into(),
        );
    }
    let expected_value_rank = match region.layout {
        pllm_core::AttentionValueQ10Layout::GroupedQuery => 4,
        pllm_core::AttentionValueQ10Layout::PerQueryHeadWindow => 5,
    };
    if region.values.shape.len() != expected_value_rank {
        return Err("semantic attention-values value rank does not match its layout".into());
    }
    let (batch, query_heads, queries, keys) = (
        region.probabilities.shape[0],
        region.probabilities.shape[1],
        region.probabilities.shape[2],
        region.probabilities.shape[3],
    );
    let consistent = match region.layout {
        pllm_core::AttentionValueQ10Layout::GroupedQuery => {
            region.values.shape[0] == batch
                && region.values.shape[2] == keys
                && query_heads % region.values.shape[1] == 0
                && region.output.shape == [batch, query_heads, queries, region.values.shape[3]]
        }
        pllm_core::AttentionValueQ10Layout::PerQueryHeadWindow => {
            region.values.shape[..4] == [batch, query_heads, queries, keys]
                && region.output.shape == [batch, query_heads, queries, region.values.shape[4]]
        }
    };
    if !consistent {
        return Err("semantic attention-values shapes do not form a valid contraction".into());
    }
    let probability_shape: [usize; 4] = region
        .probabilities
        .shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .map_err(|_| "attention-values dimension exceeds usize".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "attention-values probability shape must have rank four".to_owned())?;
    if probabilities.shape() != probability_shape {
        return Err(
            "attention-values probabilities do not match the region probability shape".into(),
        );
    }
    let value_shape: Vec<usize> = region
        .values
        .shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .map_err(|_| "attention-values dimension exceeds usize".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    pllm_core::attention_values_q10(
        probabilities.probabilities(),
        probability_shape,
        values,
        &value_shape,
        policy,
    )
    .map_err(|error| error.to_string())
}

pub fn lower_model_attention_scores_q20_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelAttentionScoresQ20Region>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::AttentionScores)
        .map(|operation| {
            let operation_id = operation.id.as_str();
            let [query_id, key_id] = operation.inputs.as_slice() else {
                return Err(format!(
                    "semantic attention-scores operation {operation_id} must have exactly two inputs"
                ));
            };
            let producer = |id: &str| {
                graph
                    .operations
                    .iter()
                    .find(|candidate| candidate.id == id)
                    .ok_or_else(|| {
                        format!(
                            "semantic attention-scores operation {operation_id} references missing input {id}"
                        )
                    })
            };
            let query = producer(query_id)?;
            if query.operator != ModelOperator::RotaryEmbedding {
                return Err(format!(
                    "semantic attention-scores operation {operation_id} must consume a rotary-embedded query"
                ));
            }
            let key = producer(key_id)?;
            let (layout, selected_positions_input_id) = match key.operator {
                ModelOperator::CacheSuffix => {
                    if key.state_kind != Some(StateKind::Key) {
                        return Err(format!(
                            "semantic attention-scores operation {operation_id} must consume the key cache view"
                        ));
                    }
                    (pllm_core::AttentionScoreQ20Layout::GroupedQueryCache, None)
                }
                ModelOperator::SecureGather => {
                    if key.state_kind != Some(StateKind::Key)
                        || key
                            .attributes
                            .get("output_semantics")
                            .and_then(serde_json::Value::as_str)
                            != Some("per_query_head_cache_window")
                    {
                        return Err(format!(
                            "semantic attention-scores operation {operation_id} must consume a per-query-head key gather"
                        ));
                    }
                    let selected = key.inputs.get(1).ok_or_else(|| {
                        format!(
                            "semantic attention-scores operation {operation_id} key gather has no selected-positions input"
                        )
                    })?;
                    (
                        pllm_core::AttentionScoreQ20Layout::PerQueryHeadWindow,
                        Some(selected.clone()),
                    )
                }
                _ => {
                    return Err(format!(
                        "semantic attention-scores operation {operation_id} must consume a key cache view or key gather"
                    ))
                }
            };
            if query.output_shape.len() != 4 || query.output_shape.contains(&0) {
                return Err(format!(
                    "semantic attention-scores operation {operation_id} requires a nonzero rank-four query shape"
                ));
            }
            let (batch, heads, queries, head_dim) = (
                query.output_shape[0],
                query.output_shape[1],
                query.output_shape[2],
                query.output_shape[3],
            );
            let group_size = operation
                .attributes
                .get("group_size")
                .and_then(serde_json::Value::as_u64)
                .filter(|group_size| *group_size > 0)
                .ok_or_else(|| {
                    format!(
                        "semantic attention-scores operation {operation_id} has invalid group_size"
                    )
                })?;
            match layout {
                pllm_core::AttentionScoreQ20Layout::GroupedQueryCache => {
                    if key.output_shape.len() != 4
                        || key.output_shape.contains(&0)
                        || key.output_shape[0] != batch
                        || key.output_shape[3] != head_dim
                        || heads % key.output_shape[1] != 0
                        || group_size != heads / key.output_shape[1]
                    {
                        return Err(format!(
                            "semantic attention-scores operation {operation_id} requires a [batch, kv_heads, keys, depth] grouped-query key view"
                        ));
                    }
                    if operation.output_shape != [batch, heads, queries, key.output_shape[2]] {
                        return Err(format!(
                            "semantic attention-scores operation {operation_id} output shape does not match the grouped-query contraction"
                        ));
                    }
                }
                pllm_core::AttentionScoreQ20Layout::PerQueryHeadWindow => {
                    if key.output_shape.len() != 5
                        || key.output_shape.contains(&0)
                        || key.output_shape[..3] != [batch, heads, queries]
                        || key.output_shape[4] != head_dim
                    {
                        return Err(format!(
                            "semantic attention-scores operation {operation_id} requires a [batch, heads, query, keys, depth] per-query-head key window"
                        ));
                    }
                    if operation.output_shape != [batch, heads, queries, key.output_shape[3]] {
                        return Err(format!(
                            "semantic attention-scores operation {operation_id} output shape does not match the per-query-head contraction"
                        ));
                    }
                }
            }
            let scales: Vec<&ModelOperation> = graph
                .operations
                .iter()
                .filter(|candidate| {
                    candidate.operator == ModelOperator::AttentionScale
                        && candidate.layer == operation.layer
                        && candidate.inputs.as_slice() == [operation.id.as_str()]
                })
                .collect();
            let [scale] = scales.as_slice() else {
                return Err(format!(
                    "semantic attention-scores operation {operation_id} requires exactly one same-layer attention scale"
                ));
            };
            if scale.output_shape != operation.output_shape {
                return Err(format!(
                    "semantic attention scale {} must preserve the score shape",
                    scale.id
                ));
            }
            let scale_head_dim = scale
                .attributes
                .get("head_dim")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    format!("semantic attention scale {} has invalid head_dim", scale.id)
                })?;
            if scale_head_dim != head_dim {
                return Err(format!(
                    "semantic attention scale {} head_dim does not match the query depth",
                    scale.id
                ));
            }
            let masks: Vec<&ModelOperation> = graph
                .operations
                .iter()
                .filter(|candidate| {
                    candidate.operator == ModelOperator::CausalMask
                        && candidate.layer == operation.layer
                        && candidate.inputs.first().map(String::as_str)
                            == Some(scale.id.as_str())
                })
                .collect();
            let [mask] = masks.as_slice() else {
                return Err(format!(
                    "semantic attention scale {} requires exactly one same-layer causal mask",
                    scale.id
                ));
            };
            if mask.output_shape != operation.output_shape {
                return Err(format!(
                    "semantic causal mask {} must preserve the score shape",
                    mask.id
                ));
            }
            let expected_mask_inputs: Vec<String> = match &selected_positions_input_id {
                None => vec![scale.id.clone(), "input.positions".to_owned()],
                Some(selected) => vec![
                    scale.id.clone(),
                    "input.positions".to_owned(),
                    selected.clone(),
                ],
            };
            if mask.inputs != expected_mask_inputs {
                return Err(format!(
                    "semantic causal mask {} has invalid attention-mask inputs",
                    mask.id
                ));
            }
            tensor_elements(&query.output_shape)?;
            tensor_elements(&key.output_shape)?;
            tensor_elements(&operation.output_shape)?;
            Ok(ModelAttentionScoresQ20Region {
                mode,
                layer: operation.layer,
                score_operation_id: operation.id.clone(),
                scale_operation_id: scale.id.clone(),
                mask_operation_id: mask.id.clone(),
                query_input_id: query_id.clone(),
                key_input_id: key_id.clone(),
                selected_positions_input_id,
                numeric_profile: pllm_core::ATTENTION_SCORE_Q20_PROFILE.to_owned(),
                layout,
                query: TensorType {
                    numeric: NumericType::SignedFixedQ10,
                    shape: query.output_shape.clone(),
                },
                key: TensorType {
                    numeric: NumericType::SignedFixedQ10,
                    shape: key.output_shape.clone(),
                },
                output: TensorType {
                    numeric: NumericType::SignedFixedQ20,
                    shape: operation.output_shape.clone(),
                },
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn execute_model_attention_scores_q20(
    region: &ModelAttentionScoresQ20Region,
    query: &[i16],
    key: &[i16],
    selected_positions: Option<&[u32]>,
    positions: &[u32],
    query_mask: &[u8],
    valid_lengths: Option<&[usize]>,
    policy: pllm_core::AttentionScoreQ20Policy,
) -> Result<pllm_core::AttentionScoresQ20, String> {
    if region.numeric_profile != pllm_core::ATTENTION_SCORE_Q20_PROFILE
        || region.query.numeric != NumericType::SignedFixedQ10
        || region.key.numeric != NumericType::SignedFixedQ10
        || region.output.numeric != NumericType::SignedFixedQ20
        || region.query.shape.len() != 4
        || region.query.shape.contains(&0)
        || region.output.shape.len() != 4
        || region.output.shape.contains(&0)
    {
        return Err(
            "semantic attention-scores requires the q10-dot-q20 profile and nonzero rank-four query/output shapes"
                .into(),
        );
    }
    let (batch, heads, queries, head_dim) = (
        region.query.shape[0],
        region.query.shape[1],
        region.query.shape[2],
        region.query.shape[3],
    );
    let query_shape: [usize; 4] = region
        .query
        .shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .map_err(|_| "attention-scores dimension exceeds usize".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "attention-scores query shape must have rank four".to_owned())?;
    match region.layout {
        pllm_core::AttentionScoreQ20Layout::GroupedQueryCache => {
            if selected_positions.is_some()
                || valid_lengths.is_none()
                || region.selected_positions_input_id.is_some()
            {
                return Err(
                    "semantic grouped-query attention-scores requires valid lengths and no selected positions"
                        .into(),
                );
            }
            if region.key.shape.len() != 4
                || region.key.shape.contains(&0)
                || region.key.shape[0] != batch
                || region.key.shape[3] != head_dim
                || heads % region.key.shape[1] != 0
                || region.output.shape != [batch, heads, queries, region.key.shape[2]]
            {
                return Err(
                    "semantic grouped-query attention-scores shapes do not form a valid contraction"
                        .into(),
                );
            }
            let key_shape: [usize; 4] = region
                .key
                .shape
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension)
                        .map_err(|_| "attention-scores dimension exceeds usize".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| "attention-scores key shape must have rank four".to_owned())?;
            pllm_core::attention_scores_q20(
                query,
                query_shape,
                key,
                key_shape,
                positions,
                query_mask,
                valid_lengths.expect("checked above"),
                policy,
            )
            .map_err(|error| error.to_string())
        }
        pllm_core::AttentionScoreQ20Layout::PerQueryHeadWindow => {
            if selected_positions.is_none()
                || valid_lengths.is_some()
                || region.selected_positions_input_id.is_none()
            {
                return Err(
                    "semantic per-query-head attention-scores requires selected positions and no valid lengths"
                        .into(),
                );
            }
            if region.key.shape.len() != 5
                || region.key.shape.contains(&0)
                || region.key.shape[..3] != [batch, heads, queries]
                || region.key.shape[4] != head_dim
                || region.output.shape != [batch, heads, queries, region.key.shape[3]]
            {
                return Err(
                    "semantic per-query-head attention-scores shapes do not form a valid window"
                        .into(),
                );
            }
            let key_shape: [usize; 5] = region
                .key
                .shape
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension)
                        .map_err(|_| "attention-scores dimension exceeds usize".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| "attention-scores key shape must have rank five".to_owned())?;
            pllm_core::attention_scores_window_q20(
                query,
                query_shape,
                key,
                key_shape,
                selected_positions.expect("checked above"),
                positions,
                query_mask,
                policy,
            )
            .map_err(|error| error.to_string())
        }
    }
}

/// Extract direct-weight FP32 RMSNorm operations into plan-bound clear reference regions.
pub fn lower_rms_norm_f32_direct_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelRmsNormF32Region>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    let plan_digest = plan.digest();
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::RmsNorm)
        .map(|operation| lower_rms_norm_f32_direct_region(&plan_digest, graph, mode, operation))
        .collect()
}

/// Execute one clear FP32 reference region after revalidating exact plan provenance.
pub fn execute_rms_norm_f32_direct(
    plan: &DecoderPlan,
    region: &ModelRmsNormF32Region,
    input: &[f32],
    weight: &[f32],
) -> Result<Vec<f32>, String> {
    if region.schema_version != RMS_NORM_F32_DIRECT_REGION_SCHEMA_VERSION
        || region.numeric_profile != pllm_core::rms_norm::RMS_NORM_F32_DIRECT_PROFILE
        || region.kernel_artifact_digest != rms_norm_kernel_artifact_digest()
        || region.weight_policy != RmsNormWeightPolicy::Direct
        || region.reduction_order != RmsNormReductionOrder::LastAxisScalarLeftToRight
    {
        return Err("direct-weight FP32 RMSNorm region contract is unsupported".into());
    }
    let authentic = lower_rms_norm_f32_direct_regions(plan, region.mode)?
        .into_iter()
        .find(|candidate| candidate.operation_id == region.operation_id)
        .ok_or("direct-weight FP32 RMSNorm region is absent from its decoder plan")?;
    if authentic != *region {
        return Err("direct-weight FP32 RMSNorm region differs from its decoder plan".into());
    }
    let width = usize::try_from(*region.shape.last().ok_or("RMSNorm shape is empty")?)
        .map_err(|_| "RMSNorm width exceeds usize")?;
    if weight.len() != width {
        return Err(format!(
            "direct-weight FP32 RMSNorm requires {width} weights, received {}",
            weight.len()
        ));
    }
    let elements = tensor_elements(&region.shape)?;
    if input.len() != elements {
        return Err(format!(
            "direct-weight FP32 RMSNorm requires {elements} input elements, received {}",
            input.len()
        ));
    }
    let epsilon = region
        .epsilon
        .parse::<f32>()
        .map_err(|_| "direct-weight FP32 RMSNorm epsilon is not an FP32 decimal")?;
    pllm_core::rms_norm_f32_direct(input, weight, epsilon).map_err(|error| error.to_string())
}

/// Extract exact-epsilon direct-weight RMSNorm operations into bounded Q10 regions.
pub fn lower_rms_norm_q10_direct_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelRmsNormQ10DirectRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    let plan_digest = plan.digest();
    graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::RmsNorm)
        .map(|operation| lower_rms_norm_q10_direct_region(&plan_digest, graph, mode, operation))
        .collect()
}

/// Execute the bounded Q10 arithmetic profile after exact plan revalidation.
pub fn execute_rms_norm_q10_direct(
    plan: &DecoderPlan,
    region: &ModelRmsNormQ10DirectRegion,
    input: &[i16],
    weight: &[i16],
) -> Result<Vec<i32>, String> {
    let (_, width) = validate_rms_norm_q10_direct_region(plan, region)?;
    if weight.len() != width {
        return Err(format!(
            "direct-weight Q10 RMSNorm requires {width} weights, received {}",
            weight.len()
        ));
    }
    let elements = tensor_elements(&region.shape)?;
    if input.len() != elements {
        return Err(format!(
            "direct-weight Q10 RMSNorm requires {elements} input elements, received {}",
            input.len()
        ));
    }
    pllm_core::rms_norm_q10_direct(input, weight).map_err(|error| error.to_string())
}

pub(crate) fn validate_rms_norm_q10_direct_region(
    plan: &DecoderPlan,
    region: &ModelRmsNormQ10DirectRegion,
) -> Result<(usize, usize), String> {
    if region.schema_version != RMS_NORM_Q10_DIRECT_REGION_SCHEMA_VERSION
        || region.numeric_profile != pllm_core::rms_norm::RMS_NORM_Q10_DIRECT_PROFILE
        || region.kernel_artifact_digest != rms_norm_kernel_artifact_digest()
        || region.epsilon_numerator != pllm_core::rms_norm::RMS_NORM_EPSILON_NUMERATOR
        || region.epsilon_denominator != pllm_core::rms_norm::RMS_NORM_EPSILON_DENOMINATOR
        || region.input_fractional_bits != 10
        || region.weight_fractional_bits != 10
        || region.reciprocal_fractional_bits != 30
        || region.output_fractional_bits != 10
        || region.maximum_width != pllm_core::rms_norm::RMS_NORM_Q10_MAX_WIDTH
        || region.maximum_encoded_error != 1
        || region.weight_policy != RmsNormWeightPolicy::Direct
        || region.reduction_order != RmsNormReductionOrder::LastAxisScalarLeftToRight
        || region.rounding != FixedPointRounding::TiesToEven
    {
        return Err("direct-weight Q10 RMSNorm region contract is unsupported".into());
    }
    let authentic = lower_rms_norm_q10_direct_regions(plan, region.mode)?
        .into_iter()
        .find(|candidate| candidate.operation_id == region.operation_id)
        .ok_or("direct-weight Q10 RMSNorm region is absent from its decoder plan")?;
    if authentic != *region {
        return Err("direct-weight Q10 RMSNorm region differs from its decoder plan".into());
    }
    let width = usize::try_from(*region.shape.last().ok_or("RMSNorm shape is empty")?)
        .map_err(|_| "RMSNorm width exceeds usize")?;
    let elements = tensor_elements(&region.shape)?;
    let rows = elements
        .checked_div(width)
        .ok_or("direct-weight Q10 RMSNorm width is zero")?;
    Ok((rows, width))
}

/// Define the exact centered-wrap32 Q14 to bounded signed-Q7 conversion contract.
pub fn define_q14_to_q7_rescale_region(
    source_operation_id: &str,
    target_operation_id: &str,
    target_input_index: u32,
    shape: Vec<u64>,
) -> Result<Q14ToQ7RescaleRegion, String> {
    if !valid_identity(source_operation_id) || !valid_identity(target_operation_id) {
        return Err("Q14-to-Q7 source or target operation ID is invalid".into());
    }
    if shape.is_empty() || shape.contains(&0) {
        return Err("Q14-to-Q7 conversion requires a non-empty tensor shape".into());
    }
    tensor_elements(&shape)?;
    Ok(Q14ToQ7RescaleRegion {
        schema_version: Q14_TO_Q7_REGION_SCHEMA_VERSION.into(),
        operation_id: q14_to_q7_rescale_operation_id(
            source_operation_id,
            target_operation_id,
            target_input_index,
        ),
        source_operation_id: source_operation_id.into(),
        target_operation_id: target_operation_id.into(),
        target_input_index,
        numeric_profile: pllm_core::fixed_point::Q14_TO_Q7_PROFILE.into(),
        input: TensorType {
            numeric: NumericType::Wrap32,
            shape: shape.clone(),
        },
        output: TensorType {
            numeric: NumericType::SignedFixedQ7,
            shape,
        },
        input_fractional_bits: 14,
        output_fractional_bits: 7,
        divisor: 128,
        rounding: FixedPointRounding::TiesToEven,
        range_policy: FixedPointRangePolicy::RejectOutsideUnitInterval,
    })
}

/// Lower the Q14-to-Q7 edges required by dense gated MLPs.
pub fn lower_model_q14_to_q7_rescale_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<Q14ToQ7RescaleRegion>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    let operations = graph
        .operations
        .iter()
        .map(|operation| (operation.id.as_str(), operation))
        .collect::<BTreeMap<_, _>>();
    let mut regions = Vec::new();

    for target in &graph.operations {
        let required_input = match target.operator {
            ModelOperator::Silu => {
                let [source_id] = target.inputs.as_slice() else {
                    return Err(format!(
                        "semantic SiLU operation {} must have exactly one input",
                        target.id
                    ));
                };
                Some((source_id, 0_u32))
            }
            ModelOperator::Multiply => {
                let [activated_id, source_id] = target.inputs.as_slice() else {
                    return Err(format!(
                        "semantic multiply operation {} must have exactly two inputs",
                        target.id
                    ));
                };
                let activated = operations.get(activated_id.as_str()).ok_or_else(|| {
                    format!(
                        "semantic multiply operation {} references missing input {activated_id}",
                        target.id
                    )
                })?;
                if activated.operator != ModelOperator::Silu
                    || activated.output_shape != target.output_shape
                {
                    return Err(format!(
                        "semantic multiply operation {} requires a shape-matched SiLU first input",
                        target.id
                    ));
                }
                Some((source_id, 1_u32))
            }
            _ => None,
        };
        let Some((source_id, target_input_index)) = required_input else {
            continue;
        };
        let source = operations.get(source_id.as_str()).ok_or_else(|| {
            format!(
                "semantic operation {} references missing rescale input {source_id}",
                target.id
            )
        })?;
        if source.operator != ModelOperator::Linear || source.output_shape != target.output_shape {
            return Err(format!(
                "semantic operation {} requires a shape-matched linear Q14 input at slot {target_input_index}",
                target.id
            ));
        }
        regions.push(define_q14_to_q7_rescale_region(
            &source.id,
            &target.id,
            target_input_index,
            source.output_shape.clone(),
        )?);
    }
    Ok(regions)
}

pub fn q14_to_q7_rescale_region_digest(region: &Q14ToQ7RescaleRegion) -> Digest {
    canonical_digest(Q14_TO_Q7_REGION_SCHEMA_VERSION, region)
}

/// Center-decode wrap32 Q14 values and rescale them under the locked contract.
pub fn execute_q14_to_q7_rescale(
    region: &Q14ToQ7RescaleRegion,
    input: &[u32],
) -> Result<Vec<i16>, String> {
    validate_q14_to_q7_rescale_region(region)?;
    let elements = tensor_elements(&region.input.shape)?;
    if input.len() != elements {
        return Err(format!(
            "Q14-to-Q7 conversion requires {elements} elements, received {}",
            input.len()
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| "Q14-to-Q7 output allocation failed")?;
    for value in input {
        output
            .push(pllm_core::rescale_q14_to_q7(*value as i32).map_err(|error| error.to_string())?);
    }
    Ok(output)
}

fn validate_q14_to_q7_rescale_region(region: &Q14ToQ7RescaleRegion) -> Result<(), String> {
    if region.schema_version != Q14_TO_Q7_REGION_SCHEMA_VERSION
        || !valid_identity(&region.source_operation_id)
        || !valid_identity(&region.target_operation_id)
        || region.operation_id
            != q14_to_q7_rescale_operation_id(
                &region.source_operation_id,
                &region.target_operation_id,
                region.target_input_index,
            )
        || region.numeric_profile != pllm_core::fixed_point::Q14_TO_Q7_PROFILE
        || region.input.numeric != NumericType::Wrap32
        || region.output.numeric != NumericType::SignedFixedQ7
        || region.input.shape != region.output.shape
        || region.input.shape.is_empty()
        || region.input.shape.contains(&0)
        || region.input_fractional_bits != 14
        || region.output_fractional_bits != 7
        || region.divisor != 128
        || region.rounding != FixedPointRounding::TiesToEven
        || region.range_policy != FixedPointRangePolicy::RejectOutsideUnitInterval
    {
        return Err(
            "Q14-to-Q7 conversion region does not match the locked numeric contract".into(),
        );
    }
    tensor_elements(&region.input.shape)?;
    Ok(())
}

fn q14_to_q7_rescale_operation_id(
    source_operation_id: &str,
    target_operation_id: &str,
    target_input_index: u32,
) -> String {
    format!("{target_operation_id}.input.{target_input_index}.from.{source_operation_id}.q14_to_q7")
}

/// Lower dense gated-MLP `SiLU(gate_proj) * up_proj` regions without name parsing.
pub fn lower_model_gated_multiply_q7_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelGatedMultiplyQ7Region>, String> {
    lower_model_gated_multiply_q7_regions_with_components(
        plan,
        mode,
        GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
        GATED_MULTIPLY_Q7_SCALAR_SCHEDULE_COMPONENT_ID,
        1,
    )
}

/// Lower dense gated-MLP regions with explicit method and scheduler components.
pub fn lower_model_gated_multiply_q7_regions_with_components(
    plan: &DecoderPlan,
    mode: DecoderMode,
    method_component_id: &str,
    schedule_component_id: &str,
    max_tensor_elements: usize,
) -> Result<Vec<ModelGatedMultiplyQ7Region>, String> {
    validate_gated_multiply_q7_components(
        method_component_id,
        schedule_component_id,
        max_tensor_elements,
    )?;
    plan.validate().map_err(|error| error.to_string())?;
    let graph = model_graph(plan, mode);
    let operations = graph
        .operations
        .iter()
        .map(|operation| (operation.id.as_str(), operation))
        .collect::<BTreeMap<_, _>>();
    let mut regions = Vec::new();

    for multiply in graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::Multiply)
    {
        let [activated_id, up_id] = multiply.inputs.as_slice() else {
            return Err(format!(
                "semantic multiply operation {} must have exactly two inputs",
                multiply.id
            ));
        };
        let activated = operations.get(activated_id.as_str()).ok_or_else(|| {
            format!(
                "semantic multiply operation {} references missing input {activated_id}",
                multiply.id
            )
        })?;
        if activated.operator != ModelOperator::Silu {
            continue;
        }
        let [gate_id] = activated.inputs.as_slice() else {
            return Err(format!(
                "semantic SiLU operation {} must have exactly one input",
                activated.id
            ));
        };
        let gate = operations.get(gate_id.as_str()).ok_or_else(|| {
            format!(
                "semantic SiLU operation {} references missing input {gate_id}",
                activated.id
            )
        })?;
        let up = operations.get(up_id.as_str()).ok_or_else(|| {
            format!(
                "semantic multiply operation {} references missing input {up_id}",
                multiply.id
            )
        })?;
        let Some(layer) = multiply.layer else {
            return Err(format!(
                "semantic gated multiply operation {} has no layer identity",
                multiply.id
            ));
        };
        if gate.operator != ModelOperator::Linear
            || up.operator != ModelOperator::Linear
            || gate.layer != Some(layer)
            || up.layer != Some(layer)
            || activated.layer != Some(layer)
            || gate.output_shape != activated.output_shape
            || gate.output_shape != up.output_shape
            || gate.output_shape != multiply.output_shape
        {
            return Err(format!(
                "semantic gated multiply operation {} requires same-layer, shape-matched linear gate/up producers",
                multiply.id
            ));
        }
        let input = TensorType {
            numeric: NumericType::SignedFixedQ7,
            shape: multiply.output_shape.clone(),
        };
        let region = ModelGatedMultiplyQ7Region {
            schema_version: GATED_MULTIPLY_Q7_REGION_SCHEMA_VERSION.into(),
            profile: SILU_Q7_EXPERIMENT_PROFILE.into(),
            model_plan_digest: plan.digest(),
            mode,
            layer,
            gate_linear_operation_id: gate.id.clone(),
            gate_rescale: define_q14_to_q7_rescale_region(
                &gate.id,
                &activated.id,
                0,
                gate.output_shape.clone(),
            )?,
            silu_operation_id: activated.id.clone(),
            up_linear_operation_id: up.id.clone(),
            up_rescale: define_q14_to_q7_rescale_region(
                &up.id,
                &multiply.id,
                1,
                up.output_shape.clone(),
            )?,
            multiply_operation_id: multiply.id.clone(),
            input: input.clone(),
            output: input,
            numeric_graph_id: GATED_MULTIPLY_Q7_NUMERIC_GRAPH_ID.into(),
            protected_graph_id: GATED_MULTIPLY_Q7_PROTECTED_GRAPH_ID.into(),
            method_component_id: method_component_id.into(),
            schedule_component_id: schedule_component_id.into(),
            max_tensor_elements,
            method: VersionedArtifact {
                id: GATED_MULTIPLY_Q7_METHOD_ID.into(),
                version: "1".into(),
                digest: gated_multiply_q7_method_artifact_digest(),
            },
            kernel: VersionedArtifact {
                id: GATED_MULTIPLY_Q7_KERNEL_ID.into(),
                version: "1".into(),
                digest: gated_multiply_q7_kernel_artifact_digest(),
            },
            compiler: VersionedArtifact {
                id: SILU_Q7_COMPILER_ID.into(),
                version: env!("CARGO_PKG_VERSION").into(),
                digest: silu_q7_compiler_artifact_digest(),
            },
        };
        validate_model_gated_multiply_q7_region(&region)?;
        regions.push(region);
    }
    Ok(regions)
}

pub fn model_gated_multiply_q7_region_digest(region: &ModelGatedMultiplyQ7Region) -> Digest {
    canonical_digest(GATED_MULTIPLY_Q7_REGION_SCHEMA_VERSION, region)
}

fn validate_model_gated_multiply_q7_region(
    region: &ModelGatedMultiplyQ7Region,
) -> Result<(), String> {
    validate_q14_to_q7_rescale_region(&region.gate_rescale)?;
    validate_q14_to_q7_rescale_region(&region.up_rescale)?;
    validate_gated_multiply_q7_components(
        &region.method_component_id,
        &region.schedule_component_id,
        region.max_tensor_elements,
    )?;
    if region.schema_version != GATED_MULTIPLY_Q7_REGION_SCHEMA_VERSION
        || region.profile != SILU_Q7_EXPERIMENT_PROFILE
        || !valid_identity(&region.gate_linear_operation_id)
        || !valid_identity(&region.silu_operation_id)
        || !valid_identity(&region.up_linear_operation_id)
        || !valid_identity(&region.multiply_operation_id)
        || region.gate_rescale.source_operation_id != region.gate_linear_operation_id
        || region.gate_rescale.target_operation_id != region.silu_operation_id
        || region.gate_rescale.target_input_index != 0
        || region.up_rescale.source_operation_id != region.up_linear_operation_id
        || region.up_rescale.target_operation_id != region.multiply_operation_id
        || region.up_rescale.target_input_index != 1
        || region.gate_rescale.output != region.input
        || region.up_rescale.output != region.input
        || region.input != region.output
        || region.input.numeric != NumericType::SignedFixedQ7
        || region.input.shape.is_empty()
        || region.input.shape.contains(&0)
        || region.numeric_graph_id != GATED_MULTIPLY_Q7_NUMERIC_GRAPH_ID
        || region.protected_graph_id != GATED_MULTIPLY_Q7_PROTECTED_GRAPH_ID
        || region.method.id != GATED_MULTIPLY_Q7_METHOD_ID
        || region.method.version != "1"
        || region.method.digest != gated_multiply_q7_method_artifact_digest()
        || region.kernel.id != GATED_MULTIPLY_Q7_KERNEL_ID
        || region.kernel.version != "1"
        || region.kernel.digest != gated_multiply_q7_kernel_artifact_digest()
        || region.compiler.id != SILU_Q7_COMPILER_ID
        || region.compiler.version != env!("CARGO_PKG_VERSION")
        || region.compiler.digest != silu_q7_compiler_artifact_digest()
    {
        return Err("gated Q7 multiply region does not match the installed contract".into());
    }
    tensor_elements(&region.input.shape)?;
    Ok(())
}

fn validate_gated_multiply_q7_components(
    method_component_id: &str,
    schedule_component_id: &str,
    max_tensor_elements: usize,
) -> Result<(), String> {
    if !matches!(
        method_component_id,
        GATED_MULTIPLY_Q7_BINARY_TABLE_COMPONENT_ID | GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID
    ) {
        return Err("unsupported gated Q7 multiply method implementation".into());
    }
    match schedule_component_id {
        GATED_MULTIPLY_Q7_SCALAR_SCHEDULE_COMPONENT_ID if max_tensor_elements == 1 => Ok(()),
        GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID
            if (2..=GATED_MULTIPLY_Q7_MAX_TENSOR_ELEMENTS).contains(&max_tensor_elements) =>
        {
            Ok(())
        }
        GATED_MULTIPLY_Q7_SCALAR_SCHEDULE_COMPONENT_ID => {
            Err("scalar gated Q7 multiply scheduling requires one element".into())
        }
        GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID => Err(format!(
            "independent-lane gated Q7 multiply scheduling supports 2..={GATED_MULTIPLY_Q7_MAX_TENSOR_ELEMENTS} elements"
        )),
        GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID
            if (1..=GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS)
                .contains(&max_tensor_elements) =>
        {
            Ok(())
        }
        GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID => Err(format!(
            "chunked independent-lane gated Q7 multiply scheduling supports 1..={GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS} elements"
        )),
        _ => Err("unsupported gated Q7 multiply schedule implementation".into()),
    }
}

fn model_graph(plan: &DecoderPlan, mode: DecoderMode) -> &DecoderGraph {
    match mode {
        DecoderMode::Prefill => &plan.prefill,
        DecoderMode::Decode => &plan.decode,
    }
}

fn lower_model_linear_region(
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelLinearRegion, String> {
    if operation.operator != ModelOperator::Linear {
        return Err(format!(
            "semantic operation {} is {:?}, not linear",
            operation.id, operation.operator
        ));
    }
    let operation_id = operation.id.as_str();
    let [input_id] = operation.inputs.as_slice() else {
        return Err(format!(
            "semantic linear operation {operation_id} must have exactly one input"
        ));
    };
    let input = graph
        .operations
        .iter()
        .find(|candidate| candidate.id == *input_id)
        .ok_or_else(|| {
            format!("semantic linear operation {operation_id} references missing input {input_id}")
        })?;
    if input.output_shape.len() < 2
        || operation.output_shape.len() != input.output_shape.len()
        || input.output_shape.contains(&0)
        || operation.output_shape.contains(&0)
        || input.output_shape[..input.output_shape.len() - 1]
            != operation.output_shape[..operation.output_shape.len() - 1]
    {
        return Err(format!(
            "semantic linear operation {operation_id} requires matching nonzero leading dimensions and one feature dimension"
        ));
    }
    let rows = input.output_shape[..input.output_shape.len() - 1]
        .iter()
        .try_fold(1_u64, |rows, dimension| {
            rows.checked_mul(*dimension)
                .ok_or("semantic linear row count overflows u64")
        })?;
    let input_tensor = TensorType {
        numeric: NumericType::Wrap32,
        shape: vec![rows, *input.output_shape.last().unwrap()],
    };
    let output_tensor = TensorType {
        numeric: NumericType::Wrap32,
        shape: vec![rows, *operation.output_shape.last().unwrap()],
    };
    let weight_id = operation
        .attributes
        .get("weight")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("semantic linear operation {operation_id} has no weight identity"))?
        .to_owned();
    let bias_id = match operation.attributes.get("bias") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(value)) if !value.is_empty() => Some(value.clone()),
        _ => {
            return Err(format!(
                "semantic linear operation {operation_id} has an invalid bias identity"
            ))
        }
    };
    Ok(ModelLinearRegion {
        mode,
        layer: operation.layer,
        operation_id: operation.id.clone(),
        input_id: input_id.clone(),
        weight_id,
        bias_id,
        input: input_tensor,
        operation: LogicalOperation {
            id: operation.id.clone(),
            operator: Operator::Linear,
            output: output_tensor,
            input_representation: Representation::MaskedRing,
            output_representation: Representation::MaskedRing,
        },
    })
}

fn lower_rms_norm_f32_direct_region(
    plan_digest: &Digest,
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelRmsNormF32Region, String> {
    let operation_id = operation.id.as_str();
    let [input_id] = operation.inputs.as_slice() else {
        return Err(format!(
            "semantic RMSNorm operation {operation_id} must have exactly one input"
        ));
    };
    let input = graph
        .operations
        .iter()
        .find(|candidate| candidate.id == *input_id)
        .ok_or_else(|| {
            format!("semantic RMSNorm operation {operation_id} references missing input {input_id}")
        })?;
    if operation.operator != ModelOperator::RmsNorm
        || input.output_shape != operation.output_shape
        || operation.output_shape.is_empty()
        || operation.output_shape.contains(&0)
    {
        return Err(format!(
            "semantic RMSNorm operation {operation_id} requires equal nonempty input and output shapes"
        ));
    }
    tensor_elements(&operation.output_shape)?;
    let weight_id = operation
        .attributes
        .get("weight")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("semantic RMSNorm operation {operation_id} has no weight identity"))?
        .to_owned();
    let weight_offset = operation
        .attributes
        .get("weight_offset")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            format!("semantic RMSNorm operation {operation_id} has no integer weight offset")
        })?;
    if weight_offset != 0 {
        return Err(format!(
            "semantic RMSNorm operation {operation_id} uses unsupported weight offset {weight_offset}"
        ));
    }
    let epsilon = operation
        .attributes
        .get("epsilon")
        .and_then(serde_json::Value::as_str)
        .filter(|value| {
            value
                .parse::<f32>()
                .is_ok_and(|epsilon| epsilon.is_finite() && epsilon > 0.0)
        })
        .ok_or_else(|| format!("semantic RMSNorm operation {operation_id} has invalid epsilon"))?
        .to_owned();
    Ok(ModelRmsNormF32Region {
        schema_version: RMS_NORM_F32_DIRECT_REGION_SCHEMA_VERSION.into(),
        model_plan_digest: plan_digest.clone(),
        mode,
        layer: operation.layer,
        operation_id: operation.id.clone(),
        input_id: input_id.clone(),
        weight_id,
        epsilon,
        shape: operation.output_shape.clone(),
        kernel_artifact_digest: rms_norm_kernel_artifact_digest(),
        numeric_profile: pllm_core::rms_norm::RMS_NORM_F32_DIRECT_PROFILE.into(),
        weight_policy: RmsNormWeightPolicy::Direct,
        reduction_order: RmsNormReductionOrder::LastAxisScalarLeftToRight,
    })
}

fn lower_rms_norm_q10_direct_region(
    plan_digest: &Digest,
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelRmsNormQ10DirectRegion, String> {
    let reference = lower_rms_norm_f32_direct_region(plan_digest, graph, mode, operation)?;
    if !decimal_equals_one_millionth(&reference.epsilon) {
        return Err(format!(
            "semantic RMSNorm operation {} epsilon {} is unsupported by the exact 1/1000000 Q10 profile",
            reference.operation_id, reference.epsilon
        ));
    }
    let width = usize::try_from(*reference.shape.last().ok_or("RMSNorm shape is empty")?)
        .map_err(|_| "RMSNorm width exceeds usize")?;
    if width > pllm_core::rms_norm::RMS_NORM_Q10_MAX_WIDTH {
        return Err(format!(
            "semantic RMSNorm operation {} width {width} exceeds Q10 profile maximum {}",
            reference.operation_id,
            pllm_core::rms_norm::RMS_NORM_Q10_MAX_WIDTH
        ));
    }
    Ok(ModelRmsNormQ10DirectRegion {
        schema_version: RMS_NORM_Q10_DIRECT_REGION_SCHEMA_VERSION.into(),
        model_plan_digest: reference.model_plan_digest,
        mode: reference.mode,
        layer: reference.layer,
        operation_id: reference.operation_id,
        input_id: reference.input_id,
        weight_id: reference.weight_id,
        epsilon_numerator: pllm_core::rms_norm::RMS_NORM_EPSILON_NUMERATOR,
        epsilon_denominator: pllm_core::rms_norm::RMS_NORM_EPSILON_DENOMINATOR,
        shape: reference.shape,
        kernel_artifact_digest: rms_norm_kernel_artifact_digest(),
        numeric_profile: pllm_core::rms_norm::RMS_NORM_Q10_DIRECT_PROFILE.into(),
        input_fractional_bits: 10,
        weight_fractional_bits: 10,
        reciprocal_fractional_bits: 30,
        output_fractional_bits: 10,
        maximum_width: pllm_core::rms_norm::RMS_NORM_Q10_MAX_WIDTH,
        maximum_encoded_error: 1,
        weight_policy: RmsNormWeightPolicy::Direct,
        reduction_order: RmsNormReductionOrder::LastAxisScalarLeftToRight,
        rounding: FixedPointRounding::TiesToEven,
    })
}

fn decimal_equals_one_millionth(value: &str) -> bool {
    let (coefficient, exponent) = value
        .split_once(['e', 'E'])
        .map_or((value, "0"), |parts| parts);
    let Ok(exponent) = exponent.parse::<i32>() else {
        return false;
    };
    let (integer, fraction) = coefficient
        .split_once('.')
        .map_or((coefficient, ""), |parts| parts);
    if integer.starts_with('-')
        || integer.starts_with('+')
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        || integer.is_empty()
        || integer.len() + fraction.len() > 38
    {
        return false;
    }
    let digits = format!("{integer}{fraction}");
    let Ok(numerator) = digits.parse::<u128>() else {
        return false;
    };
    let Ok(fraction_digits) = i32::try_from(fraction.len()) else {
        return false;
    };
    let scale = fraction_digits - exponent;
    if !(0..=38).contains(&scale) {
        return false;
    }
    numerator.checked_mul(1_000_000)
        == 10_u128.checked_pow(u32::try_from(scale).expect("nonnegative bounded scale"))
}

fn lower_model_residual_region(
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelResidualRegion, String> {
    let operation_id = operation.id.as_str();
    let [left_id, right_id] = operation.inputs.as_slice() else {
        return Err(format!(
            "semantic residual operation {operation_id} must have exactly two inputs"
        ));
    };
    let input_shape = [left_id, right_id]
        .map(|input_id| {
            graph
                .operations
                .iter()
                .find(|candidate| candidate.id == *input_id)
                .map(|input| input.output_shape.as_slice())
                .ok_or_else(|| {
                    format!(
                        "semantic residual operation {operation_id} references missing input {input_id}"
                    )
                })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    if input_shape[0] != input_shape[1] || input_shape[0] != operation.output_shape {
        return Err(format!(
            "semantic residual operation {operation_id} requires equal input and output shapes"
        ));
    }
    tensor_elements(&operation.output_shape)?;
    let tensor = TensorType {
        numeric: NumericType::Wrap32,
        shape: operation.output_shape.clone(),
    };
    Ok(ModelResidualRegion {
        mode,
        layer: operation.layer,
        operation_id: operation.id.clone(),
        input_ids: [left_id.clone(), right_id.clone()],
        input: tensor.clone(),
        output: tensor,
    })
}

fn lower_model_output_head_region(
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelOutputHeadRegion, String> {
    let operation_id = operation.id.as_str();
    let [input_id] = operation.inputs.as_slice() else {
        return Err(format!(
            "semantic output-head operation {operation_id} must have exactly one input"
        ));
    };
    let input = graph
        .operations
        .iter()
        .find(|candidate| candidate.id == *input_id)
        .ok_or_else(|| {
            format!(
                "semantic output-head operation {operation_id} references missing input {input_id}"
            )
        })?;
    if operation.operator != ModelOperator::OutputHead
        || input.output_shape.len() < 2
        || operation.output_shape.len() != input.output_shape.len()
        || input.output_shape.contains(&0)
        || operation.output_shape.contains(&0)
        || input.output_shape[..input.output_shape.len() - 1]
            != operation.output_shape[..operation.output_shape.len() - 1]
    {
        return Err(format!(
            "semantic output-head operation {operation_id} requires matching nonzero leading dimensions and one feature dimension"
        ));
    }
    tensor_elements(&input.output_shape)?;
    tensor_elements(&operation.output_shape)?;
    let weight_id = operation
        .attributes
        .get("weight")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            format!("semantic output-head operation {operation_id} has no weight identity")
        })?
        .to_owned();
    Ok(ModelOutputHeadRegion {
        mode,
        layer: operation.layer,
        operation_id: operation.id.clone(),
        input_id: input_id.clone(),
        weight_id,
        input: TensorType {
            numeric: NumericType::Wrap32,
            shape: input.output_shape.clone(),
        },
        output: TensorType {
            numeric: NumericType::Wrap32,
            shape: operation.output_shape.clone(),
        },
    })
}

fn lower_model_last_token_region(
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelLastTokenRegion, String> {
    let operation_id = operation.id.as_str();
    let selection = operation
        .attributes
        .get("selection")
        .and_then(serde_json::Value::as_str);
    let valid_lengths_input = operation
        .attributes
        .get("valid_lengths_input")
        .and_then(serde_json::Value::as_str);
    let (input_id, selection_kind, sequence_lengths_input_id) = match operation.inputs.as_slice() {
        [input_id] if selection.is_none() && valid_lengths_input.is_none() => {
            (input_id, ModelLastTokenSelection::PhysicalLast, None)
        }
        [input_id, lengths]
            if lengths == "input.sequence_lengths"
                && selection == Some("last_valid")
                && valid_lengths_input == Some("input.sequence_lengths") =>
        {
            (
                input_id,
                ModelLastTokenSelection::LastValid,
                Some("input.sequence_lengths".to_owned()),
            )
        }
        _ => {
            return Err(format!(
                "semantic last-token operation {operation_id} has contradictory selection metadata"
            ))
        }
    };
    let input = graph
        .operations
        .iter()
        .find(|candidate| candidate.id == *input_id)
        .ok_or_else(|| {
            format!(
                "semantic last-token operation {operation_id} references missing input {input_id}"
            )
        })?;
    let axis = operation
        .attributes
        .get("axis")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| format!("semantic last-token operation {operation_id} has no valid axis"))?;
    let rank = i64::try_from(input.output_shape.len())
        .map_err(|_| format!("semantic last-token operation {operation_id} rank overflowed"))?;
    if rank < 2 || axis < -rank || axis >= rank {
        return Err(format!(
            "semantic last-token operation {operation_id} axis is out of range"
        ));
    }
    let axis = usize::try_from(if axis < 0 { rank + axis } else { axis })
        .map_err(|_| format!("semantic last-token operation {operation_id} axis is invalid"))?;
    let mut expected_output_shape = input.output_shape.clone();
    expected_output_shape.remove(axis);
    if operation.output_shape != expected_output_shape {
        return Err(format!(
            "semantic last-token operation {operation_id} output shape does not remove its selected axis"
        ));
    }
    tensor_elements(&input.output_shape)?;
    tensor_elements(&operation.output_shape)?;
    Ok(ModelLastTokenRegion {
        mode,
        layer: operation.layer,
        operation_id: operation.id.clone(),
        input_id: input_id.clone(),
        sequence_lengths_input_id,
        selection: selection_kind,
        axis,
        input: TensorType {
            numeric: NumericType::Wrap32,
            shape: input.output_shape.clone(),
        },
        output: TensorType {
            numeric: NumericType::Wrap32,
            shape: operation.output_shape.clone(),
        },
    })
}

fn lower_model_reshape_region(
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelReshapeRegion, String> {
    let operation_id = operation.id.as_str();
    let [input_id] = operation.inputs.as_slice() else {
        return Err(format!(
            "semantic reshape operation {operation_id} must have exactly one input"
        ));
    };
    let input = graph
        .operations
        .iter()
        .find(|candidate| candidate.id == *input_id)
        .ok_or_else(|| {
            format!("semantic reshape operation {operation_id} references missing input {input_id}")
        })?;
    let layout = match operation
        .attributes
        .get("layout")
        .and_then(serde_json::Value::as_str)
    {
        Some("batch_heads_sequence_feature") => ModelReshapeLayout::BatchHeadsSequenceFeature,
        Some("batch_sequence_hidden") => ModelReshapeLayout::BatchSequenceHidden,
        _ => {
            return Err(format!(
                "semantic reshape operation {operation_id} has an unsupported layout"
            ))
        }
    };
    let region = ModelReshapeRegion {
        mode,
        layer: operation.layer,
        operation_id: operation.id.clone(),
        input_id: input_id.clone(),
        layout,
        input: TensorType {
            numeric: NumericType::Wrap32,
            shape: input.output_shape.clone(),
        },
        output: TensorType {
            numeric: NumericType::Wrap32,
            shape: operation.output_shape.clone(),
        },
    };
    validate_model_reshape_shapes(&region)?;
    Ok(region)
}

fn validate_model_reshape_shapes(region: &ModelReshapeRegion) -> Result<(), String> {
    let (batch, sequence, hidden, output_batch, output_sequence, heads, feature) = match region
        .layout
    {
        ModelReshapeLayout::BatchHeadsSequenceFeature => {
            let [batch, sequence, hidden] = region.input.shape.as_slice() else {
                return Err("batch_heads_sequence_feature requires rank-3 input".into());
            };
            let [output_batch, heads, output_sequence, feature] = region.output.shape.as_slice()
            else {
                return Err("batch_heads_sequence_feature requires rank-4 output".into());
            };
            (
                batch,
                sequence,
                hidden,
                output_batch,
                output_sequence,
                heads,
                feature,
            )
        }
        ModelReshapeLayout::BatchSequenceHidden => {
            let [batch, heads, sequence, feature] = region.input.shape.as_slice() else {
                return Err("batch_sequence_hidden requires rank-4 input".into());
            };
            let [output_batch, output_sequence, hidden] = region.output.shape.as_slice() else {
                return Err("batch_sequence_hidden requires rank-3 output".into());
            };
            (
                batch,
                sequence,
                hidden,
                output_batch,
                output_sequence,
                heads,
                feature,
            )
        }
    };
    if batch != output_batch
        || sequence != output_sequence
        || hidden
            != &heads
                .checked_mul(*feature)
                .ok_or("reshape shape overflows u64")?
    {
        return Err("semantic reshape shapes are incompatible".into());
    }
    Ok(())
}

fn tensor_elements(shape: &[u64]) -> Result<usize, String> {
    let elements = shape.iter().try_fold(1_u64, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or("semantic tensor element count overflows u64")
    })?;
    usize::try_from(elements).map_err(|_| "semantic tensor element count exceeds usize".into())
}

/// Extract one semantic SiLU operation into the compiler's locked Q7 representation.
pub fn lower_model_silu_operation(
    plan: &DecoderPlan,
    mode: DecoderMode,
    operation_id: &str,
) -> Result<(TensorType, LogicalOperation), String> {
    plan.validate().map_err(|error| error.to_string())?;
    let graph = match mode {
        DecoderMode::Prefill => &plan.prefill,
        DecoderMode::Decode => &plan.decode,
    };
    let operation = graph
        .operations
        .iter()
        .find(|operation| operation.id == operation_id)
        .ok_or_else(|| format!("semantic operation {operation_id} is absent from {mode:?}"))?;
    if operation.operator != ModelOperator::Silu {
        return Err(format!(
            "semantic operation {operation_id} is {:?}, not SiLU",
            operation.operator
        ));
    }
    let [input_id] = operation.inputs.as_slice() else {
        return Err(format!(
            "semantic SiLU operation {operation_id} must have exactly one input"
        ));
    };
    let input = graph
        .operations
        .iter()
        .find(|candidate| candidate.id == *input_id)
        .ok_or_else(|| {
            format!("semantic SiLU operation {operation_id} references missing input {input_id}")
        })?;
    if input.output_shape != operation.output_shape {
        return Err(format!(
            "semantic SiLU operation {operation_id} input and output shapes differ"
        ));
    }
    let tensor = TensorType {
        numeric: NumericType::SignedFixedQ7,
        shape: operation.output_shape.clone(),
    };
    Ok((
        tensor.clone(),
        LogicalOperation {
            id: operation.id.clone(),
            operator: Operator::Silu,
            output: tensor,
            input_representation: Representation::ArithmeticLabel,
            output_representation: Representation::ArithmeticLabel,
        },
    ))
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SiluQ7GateHeader {
    schema_version: String,
    compiled_plan_digest: Digest,
    operation_id: String,
    numeric_graph_id: String,
    protected_graph_id: String,
    gate_digest: Digest,
}

fn silu_q7_compiled_plan_digest(compiled: &CompiledPlan) -> Digest {
    canonical_digest("pllm.compiled-plan.silu-q7.v1", compiled)
}

/// Client-only encodings plus a plan-bound, serializable evaluator gate.
pub struct BoundSiluQ7Material {
    material: SiluQuadraticQ7Material,
    evaluator_payload: Vec<u8>,
}

impl BoundSiluQ7Material {
    pub fn evaluator_payload(&self) -> Vec<u8> {
        self.evaluator_payload.clone()
    }

    pub fn encode(&self, value: i16) -> Result<Vec<u8>, pllm_garble::GarbleError> {
        self.material.encode(value)
    }

    pub fn decode(&self, bytes: &[u8]) -> Result<i16, pllm_garble::GarbleError> {
        self.material.decode(bytes)
    }
}

/// Prepare one gate committed to the exact compiled execution plan and operation.
pub fn prepare_bound_silu_q7_material(
    compiled: &CompiledPlan,
) -> Result<BoundSiluQ7Material, String> {
    let (_, step) = validate_silu_q7_plan(compiled)?;
    let plan_digest = silu_q7_compiled_plan_digest(compiled);
    let material = pllm_garble::prepare_silu_quadratic_q7_with_context(digest_array(&plan_digest)?)
        .map_err(|error| error.to_string())?;
    let gate = material.gate_bytes();
    let header = SiluQ7GateHeader {
        schema_version: SILU_Q7_GATE_SCHEMA_VERSION.into(),
        compiled_plan_digest: plan_digest,
        operation_id: step.operation_id.clone(),
        numeric_graph_id: SILU_Q7_NUMERIC_GRAPH_ID.into(),
        protected_graph_id: SILU_Q7_PROTECTED_GRAPH_ID.into(),
        gate_digest: digest_bytes(SILU_Q7_GATE_SCHEMA_VERSION, &gate),
    };
    let encoded_header = canonical_bytes(&header);
    let header_len = u32::try_from(encoded_header.len())
        .map_err(|_| "Q7 SiLU gate header exceeds u32".to_owned())?;
    let mut evaluator_payload = Vec::with_capacity(4 + encoded_header.len() + gate.len());
    evaluator_payload.extend_from_slice(&header_len.to_le_bytes());
    evaluator_payload.extend_from_slice(&encoded_header);
    evaluator_payload.extend_from_slice(&gate);
    let gate =
        pllm_garble::GarbledProjection::from_bytes(&gate).map_err(|error| error.to_string())?;
    register_silu_q7_material(
        gate.material_id(),
        digest_bytes(SILU_Q7_ISSUANCE_DIGEST_DOMAIN, &evaluator_payload),
    )?;
    Ok(BoundSiluQ7Material {
        material,
        evaluator_payload,
    })
}

static SILU_Q7_ISSUANCE_REGISTRY: OnceLock<Mutex<BTreeMap<[u8; 32], Digest>>> = OnceLock::new();

fn register_silu_q7_material(material_id: [u8; 32], payload_digest: Digest) -> Result<(), String> {
    let mut registry = SILU_Q7_ISSUANCE_REGISTRY
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "Q7 SiLU issuance registry is poisoned")?;
    if registry.contains_key(&material_id) {
        return Err("Q7 SiLU material identifier collision".into());
    }
    if registry.len() >= SILU_Q7_ISSUANCE_CAPACITY {
        return Err("Q7 SiLU issuance capacity is exhausted".into());
    }
    registry.insert(material_id, payload_digest);
    Ok(())
}

fn consume_silu_q7_material(materials: &BTreeMap<[u8; 32], Digest>) -> Result<(), String> {
    let mut registry = SILU_Q7_ISSUANCE_REGISTRY
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "Q7 SiLU issuance registry is poisoned")?;
    let all_issued = materials
        .iter()
        .all(|(id, digest)| registry.get(id) == Some(digest));
    for (id, digest) in materials {
        if registry.get(id) == Some(digest) {
            registry.remove(id);
        }
    }
    if !all_issued {
        return Err("Q7 SiLU evaluator material was not issued or was already bound".into());
    }
    Ok(())
}

/// Plan-bound evaluator state for one transported Q7 SiLU tensor.
pub struct SiluQ7Evaluator {
    gates: Vec<pllm_garble::GarbledProjection>,
    consumed: bool,
}

impl SiluQ7Evaluator {
    /// Parse, verify, and burn evaluator-only gate payloads before use.
    pub fn new(compiled: &CompiledPlan, payloads: &[Vec<u8>]) -> Result<Self, String> {
        let (elements, step) = validate_silu_q7_plan(compiled)?;
        if payloads.len() != elements {
            return Err(format!(
                "Q7 SiLU expected {elements} gates, got {}",
                payloads.len()
            ));
        }
        let expected_plan = silu_q7_compiled_plan_digest(compiled);
        let mut decoded = Vec::with_capacity(elements);
        let mut issued_materials = BTreeMap::new();
        let mut duplicate = false;
        for payload in payloads {
            let (header, gate_bytes) = decode_silu_q7_gate(payload)?;
            let gate = pllm_garble::GarbledProjection::from_bytes(gate_bytes)
                .map_err(|error| error.to_string())?;
            duplicate |= issued_materials
                .insert(
                    gate.material_id(),
                    digest_bytes(SILU_Q7_ISSUANCE_DIGEST_DOMAIN, payload),
                )
                .is_some();
            decoded.push((header, gate));
        }
        consume_silu_q7_material(&issued_materials)?;
        if duplicate {
            return Err("Q7 SiLU evaluator payload contains duplicate material".into());
        }
        let mut gates = Vec::with_capacity(elements);
        for (header, gate) in decoded {
            if header.compiled_plan_digest != expected_plan
                || header.operation_id != step.operation_id
                || header.numeric_graph_id != SILU_Q7_NUMERIC_GRAPH_ID
                || header.protected_graph_id != SILU_Q7_PROTECTED_GRAPH_ID
            {
                return Err("Q7 SiLU gate commitment does not match compiled plan".into());
            }
            if gate.context_digest() != digest_array(&expected_plan)? {
                return Err("Q7 SiLU gate authentication is bound to another plan".into());
            }
            if gate.input_moduli() != [pllm_garble::SILU_QUADRATIC_Q7_MODULUS]
                || gate.output_modulus() != pllm_garble::SILU_QUADRATIC_Q7_MODULUS
            {
                return Err("Q7 SiLU gate uses the wrong arithmetic modulus".into());
            }
            gates.push(gate);
        }
        Ok(Self {
            gates,
            consumed: false,
        })
    }

    /// Consume every bound gate on the first evaluation attempt, including failure.
    pub fn evaluate(&mut self, input_labels: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
        self.burn()?;
        if input_labels.len() != self.gates.len() {
            return Err(format!(
                "Q7 SiLU expected {} labels, got {}",
                self.gates.len(),
                input_labels.len()
            ));
        }
        std::mem::take(&mut self.gates)
            .into_iter()
            .zip(input_labels)
            .map(|(gate, label)| {
                let label =
                    pllm_garble::Label::from_bytes(label).map_err(|error| error.to_string())?;
                gate.evaluate(&[&label])
                    .map(|output| output.to_bytes())
                    .map_err(|error| error.to_string())
            })
            .collect()
    }

    /// Burn a rejected request before allocating or parsing its input labels.
    pub fn burn(&mut self) -> Result<(), String> {
        if self.consumed {
            return Err("Q7 SiLU evaluator material was already consumed".into());
        }
        self.consumed = true;
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GatedMultiplyQ7GateHeader {
    schema_version: String,
    region_digest: Digest,
    #[serde(rename = "method_implementation")]
    method_component_id: String,
    #[serde(rename = "schedule_implementation")]
    schedule_component_id: String,
    element_count: usize,
    bundle_id: Digest,
    program_digests: Vec<Digest>,
}

/// Client-only encodings plus a region-bound evaluator program.
pub struct BoundGatedMultiplyQ7Material {
    materials: Vec<pllm_garble::GatedMultiplyQ7Material>,
    evaluator_payload: Vec<u8>,
    issuance_id: [u8; 32],
    issuance_digest: Digest,
}

impl BoundGatedMultiplyQ7Material {
    pub fn evaluator_payload(&self) -> Vec<u8> {
        self.evaluator_payload.clone()
    }

    pub fn element_count(&self) -> usize {
        self.materials.len()
    }

    /// Cancel evaluator issuance when prepared material will not be used.
    pub fn cancel(&self) -> Result<bool, String> {
        unregister_gated_multiply_q7_material(self.issuance_id, &self.issuance_digest)
    }

    pub fn encode_gate(&self, value: i16) -> Result<Vec<u8>, pllm_garble::GarbleError> {
        let [material] = self.materials.as_slice() else {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        };
        material.encode_gate(value)
    }

    pub fn encode_up(&self, value: i16) -> Result<Vec<u8>, pllm_garble::GarbleError> {
        let [material] = self.materials.as_slice() else {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        };
        material.encode_up(value)
    }

    pub fn decode(&self, bytes: &[u8]) -> Result<i16, pllm_garble::GarbleError> {
        let [material] = self.materials.as_slice() else {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        };
        material.decode(bytes)
    }

    pub fn encode_gates(&self, values: &[i16]) -> Result<Vec<Vec<u8>>, pllm_garble::GarbleError> {
        if values.len() != self.materials.len() {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        }
        self.materials
            .iter()
            .zip(values)
            .map(|(material, value)| material.encode_gate(*value))
            .collect()
    }

    pub fn encode_ups(&self, values: &[i16]) -> Result<Vec<Vec<u8>>, pllm_garble::GarbleError> {
        if values.len() != self.materials.len() {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        }
        self.materials
            .iter()
            .zip(values)
            .map(|(material, value)| material.encode_up(*value))
            .collect()
    }

    pub fn decode_tensor(&self, labels: &[Vec<u8>]) -> Result<Vec<i16>, pllm_garble::GarbleError> {
        if labels.len() != self.materials.len() {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        }
        self.materials
            .iter()
            .zip(labels)
            .map(|(material, label)| material.decode(label))
            .collect()
    }
}

impl Drop for BoundGatedMultiplyQ7Material {
    fn drop(&mut self) {
        let _ = unregister_gated_multiply_q7_material(self.issuance_id, &self.issuance_digest);
    }
}

/// Prepare a bounded, one-use, label-preserving `SiLU(gate) * up` bundle.
pub fn prepare_bound_gated_multiply_q7_material(
    plan: &DecoderPlan,
    region: &ModelGatedMultiplyQ7Region,
) -> Result<BoundGatedMultiplyQ7Material, String> {
    validate_model_gated_multiply_q7_region_against_plan(plan, region)?;
    let elements = validate_schedulable_gated_multiply_q7_region(region)?;
    let region_digest = model_gated_multiply_q7_region_digest(region);
    let method = gated_multiply_q7_method(region)?;
    let mut materials = Vec::with_capacity(elements);
    let mut programs = Vec::with_capacity(elements);
    for lane in 0..elements {
        let context = gated_multiply_q7_lane_context(&region_digest, lane)?;
        let material =
            pllm_garble::prepare_gated_multiply_q7_with_method_and_context(method, context)
                .map_err(|error| error.to_string())?;
        programs.push(material.program_bytes());
        materials.push(material);
    }
    let material_ids = materials
        .iter()
        .map(pllm_garble::GatedMultiplyQ7Material::material_id)
        .collect::<Vec<_>>();
    let bundle_id = canonical_digest(GATED_MULTIPLY_Q7_BUNDLE_ID_DOMAIN, &material_ids);
    let bundle_id_bytes = digest_array(&bundle_id)?;
    let header = GatedMultiplyQ7GateHeader {
        schema_version: GATED_MULTIPLY_Q7_PROGRAM_SCHEMA_VERSION.into(),
        region_digest,
        method_component_id: region.method_component_id.clone(),
        schedule_component_id: region.schedule_component_id.clone(),
        element_count: elements,
        bundle_id,
        program_digests: programs
            .iter()
            .map(|program| digest_bytes(GATED_MULTIPLY_Q7_PROGRAM_SCHEMA_VERSION, program))
            .collect(),
    };
    let encoded_header = canonical_bytes(&header);
    let header_len = u32::try_from(encoded_header.len())
        .map_err(|_| "gated Q7 multiply header exceeds u32".to_owned())?;
    let program_bytes = programs.iter().map(Vec::len).sum::<usize>();
    let mut evaluator_payload = Vec::with_capacity(
        4 + encoded_header.len() + programs.len().saturating_mul(4) + program_bytes,
    );
    evaluator_payload.extend_from_slice(&header_len.to_le_bytes());
    evaluator_payload.extend_from_slice(&encoded_header);
    for program in programs {
        let length = u32::try_from(program.len())
            .map_err(|_| "gated Q7 multiply program exceeds u32".to_owned())?;
        evaluator_payload.extend_from_slice(&length.to_le_bytes());
        evaluator_payload.extend_from_slice(&program);
    }
    if evaluator_payload.len() > GATED_MULTIPLY_Q7_MAX_EVALUATOR_PAYLOAD_BYTES {
        return Err("gated Q7 multiply evaluator payload exceeds its byte bound".into());
    }
    let issuance_digest =
        digest_bytes(GATED_MULTIPLY_Q7_ISSUANCE_DIGEST_DOMAIN, &evaluator_payload);
    register_gated_multiply_q7_material(bundle_id_bytes, issuance_digest.clone())?;
    Ok(BoundGatedMultiplyQ7Material {
        materials,
        evaluator_payload,
        issuance_id: bundle_id_bytes,
        issuance_digest,
    })
}

fn validate_model_gated_multiply_q7_region_against_plan(
    plan: &DecoderPlan,
    region: &ModelGatedMultiplyQ7Region,
) -> Result<(), String> {
    validate_model_gated_multiply_q7_region(region)?;
    if region.model_plan_digest != plan.digest() {
        return Err("gated Q7 multiply region does not match the decoder plan digest".into());
    }
    let expected = lower_model_gated_multiply_q7_regions_with_components(
        plan,
        region.mode,
        &region.method_component_id,
        &region.schedule_component_id,
        region.max_tensor_elements,
    )?;
    if !expected.iter().any(|candidate| candidate == region) {
        return Err("gated Q7 multiply region was not lowered from the decoder plan".into());
    }
    Ok(())
}

static GATED_MULTIPLY_Q7_ISSUANCE_REGISTRY: OnceLock<Mutex<BTreeMap<[u8; 32], Digest>>> =
    OnceLock::new();

fn register_gated_multiply_q7_material(
    material_id: [u8; 32],
    payload_digest: Digest,
) -> Result<(), String> {
    let mut registry = GATED_MULTIPLY_Q7_ISSUANCE_REGISTRY
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "gated Q7 multiply issuance registry is poisoned")?;
    if registry.contains_key(&material_id) {
        return Err("gated Q7 multiply material identifier collision".into());
    }
    if registry.len() >= GATED_MULTIPLY_Q7_ISSUANCE_CAPACITY {
        return Err("gated Q7 multiply issuance capacity is exhausted".into());
    }
    registry.insert(material_id, payload_digest);
    Ok(())
}

fn unregister_gated_multiply_q7_material(
    material_id: [u8; 32],
    payload_digest: &Digest,
) -> Result<bool, String> {
    let mut registry = GATED_MULTIPLY_Q7_ISSUANCE_REGISTRY
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "gated Q7 multiply issuance registry is poisoned")?;
    if registry.get(&material_id) != Some(payload_digest) {
        return Ok(false);
    }
    registry.remove(&material_id);
    Ok(true)
}

fn consume_gated_multiply_q7_material(
    material_id: [u8; 32],
    payload_digest: &Digest,
) -> Result<(), String> {
    let mut registry = GATED_MULTIPLY_Q7_ISSUANCE_REGISTRY
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "gated Q7 multiply issuance registry is poisoned")?;
    if registry.get(&material_id) != Some(payload_digest) {
        return Err(
            "gated Q7 multiply evaluator material was not issued or was already bound".into(),
        );
    }
    registry.remove(&material_id);
    Ok(())
}

/// Bound evaluator state for one scalar or bounded tensor gated Q7 multiplication.
pub struct GatedMultiplyQ7Evaluator {
    programs: Option<Vec<pllm_garble::GarbledProgram>>,
    consumed: bool,
}

impl GatedMultiplyQ7Evaluator {
    pub fn new(region: &ModelGatedMultiplyQ7Region, payload: &[u8]) -> Result<Self, String> {
        let elements = validate_schedulable_gated_multiply_q7_region(region)?;
        let expected_region = model_gated_multiply_q7_region_digest(region);
        let (header, program_bytes) = decode_gated_multiply_q7_program(payload)?;
        consume_gated_multiply_q7_material(
            digest_array(&header.bundle_id)?,
            &digest_bytes(GATED_MULTIPLY_Q7_ISSUANCE_DIGEST_DOMAIN, payload),
        )?;
        if header.region_digest != expected_region
            || header.method_component_id != region.method_component_id
            || header.schedule_component_id != region.schedule_component_id
            || header.element_count != elements
            || header.program_digests.len() != elements
            || program_bytes.len() != elements
        {
            return Err("gated Q7 multiply commitment does not match region".into());
        }
        let mut programs = Vec::with_capacity(elements);
        for (lane, (bytes, digest)) in program_bytes
            .into_iter()
            .zip(&header.program_digests)
            .enumerate()
        {
            if digest_bytes(GATED_MULTIPLY_Q7_PROGRAM_SCHEMA_VERSION, bytes) != *digest {
                return Err("gated Q7 multiply program commitment is invalid".into());
            }
            let program = pllm_garble::GarbledProgram::from_bytes(bytes)
                .map_err(|error| error.to_string())?;
            if program.context_digest() != gated_multiply_q7_lane_context(&expected_region, lane)? {
                return Err("gated Q7 multiply authentication is bound to another lane".into());
            }
            if program.input_moduli()
                != [
                    pllm_garble::SILU_QUADRATIC_Q7_MODULUS,
                    pllm_garble::SILU_QUADRATIC_Q7_MODULUS,
                ]
            {
                return Err("gated Q7 multiply program uses the wrong arithmetic modulus".into());
            }
            programs.push(program);
        }
        Ok(Self {
            programs: Some(programs),
            consumed: false,
        })
    }

    pub fn element_count(&self) -> usize {
        self.programs.as_ref().map_or(0, Vec::len)
    }

    /// Consume the complete program on the first attempt, including malformed input.
    pub fn evaluate(&mut self, gate_label: &[u8], up_label: &[u8]) -> Result<Vec<u8>, String> {
        let mut outputs = self.evaluate_tensor(&[gate_label], &[up_label])?;
        outputs
            .pop()
            .ok_or("gated Q7 multiply scalar output is absent".into())
    }

    /// Burn every lane before label parsing and release no partial output.
    pub fn evaluate_tensor(
        &mut self,
        gate_labels: &[&[u8]],
        up_labels: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, String> {
        self.burn()?;
        let programs = self
            .programs
            .take()
            .ok_or("gated Q7 multiply evaluator material is absent")?;
        if gate_labels.len() != programs.len() || up_labels.len() != programs.len() {
            return Err("gated Q7 multiply label count does not match the bundle".into());
        }
        programs
            .into_iter()
            .zip(gate_labels.iter().zip(up_labels))
            .map(|(program, (gate, up))| {
                program
                    .evaluate(&[gate, up])
                    .map_err(|error| error.to_string())
            })
            .collect()
    }

    pub fn burn(&mut self) -> Result<(), String> {
        if self.consumed {
            return Err("gated Q7 multiply evaluator material was already consumed".into());
        }
        self.consumed = true;
        Ok(())
    }
}

fn validate_schedulable_gated_multiply_q7_region(
    region: &ModelGatedMultiplyQ7Region,
) -> Result<usize, String> {
    validate_model_gated_multiply_q7_region(region)?;
    let elements = tensor_elements(&region.input.shape)?;
    if elements > region.max_tensor_elements {
        return Err(format!(
            "gated Q7 multiply scheduler permits at most {} elements, received {elements}",
            region.max_tensor_elements
        ));
    }
    Ok(elements)
}

fn decode_gated_multiply_q7_program(
    payload: &[u8],
) -> Result<(GatedMultiplyQ7GateHeader, Vec<&[u8]>), String> {
    if payload.len() > GATED_MULTIPLY_Q7_MAX_EVALUATOR_PAYLOAD_BYTES {
        return Err("gated Q7 multiply evaluator payload exceeds its byte bound".into());
    }
    let header_len = u32::from_le_bytes(
        payload
            .get(..4)
            .ok_or("gated Q7 multiply payload is truncated")?
            .try_into()
            .map_err(|_| "gated Q7 multiply payload is truncated")?,
    ) as usize;
    let header_end = 4_usize
        .checked_add(header_len)
        .ok_or("gated Q7 multiply header length overflows")?;
    let header_bytes = payload
        .get(4..header_end)
        .ok_or("gated Q7 multiply header is truncated")?;
    let header: GatedMultiplyQ7GateHeader =
        serde_json::from_slice(header_bytes).map_err(|_| "gated Q7 multiply header is invalid")?;
    if header.schema_version != GATED_MULTIPLY_Q7_PROGRAM_SCHEMA_VERSION
        || canonical_bytes(&header) != header_bytes
        || header.element_count == 0
        || header.element_count > GATED_MULTIPLY_Q7_MAX_TENSOR_ELEMENTS
        || header.program_digests.len() != header.element_count
    {
        return Err("gated Q7 multiply program commitment is invalid".into());
    }
    let mut position = header_end;
    let mut programs = Vec::with_capacity(header.element_count);
    for _ in 0..header.element_count {
        let length_end = position
            .checked_add(4)
            .ok_or("gated Q7 multiply program length overflows")?;
        let length = u32::from_le_bytes(
            payload
                .get(position..length_end)
                .ok_or("gated Q7 multiply program length is truncated")?
                .try_into()
                .map_err(|_| "gated Q7 multiply program length is truncated")?,
        ) as usize;
        position = length_end;
        let end = position
            .checked_add(length)
            .ok_or("gated Q7 multiply program length overflows")?;
        let program = payload
            .get(position..end)
            .filter(|program| !program.is_empty())
            .ok_or("gated Q7 multiply program is truncated")?;
        programs.push(program);
        position = end;
    }
    if position != payload.len() {
        return Err("gated Q7 multiply payload has trailing bytes".into());
    }
    Ok((header, programs))
}

fn gated_multiply_q7_method(
    region: &ModelGatedMultiplyQ7Region,
) -> Result<pllm_garble::GatedMultiplyQ7Method, String> {
    match region.method_component_id.as_str() {
        GATED_MULTIPLY_Q7_BINARY_TABLE_COMPONENT_ID => {
            Ok(pllm_garble::GatedMultiplyQ7Method::BinaryTable)
        }
        GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID => Ok(pllm_garble::GatedMultiplyQ7Method::R03Crt),
        _ => Err("unsupported gated Q7 multiply method implementation".into()),
    }
}

fn gated_multiply_q7_lane_context(region_digest: &Digest, lane: usize) -> Result<[u8; 32], String> {
    digest_array(&canonical_digest(
        GATED_MULTIPLY_Q7_LANE_CONTEXT_DOMAIN,
        &(region_digest, lane),
    ))
}

fn digest_array(digest: &Digest) -> Result<[u8; 32], String> {
    let encoded = digest.as_str().as_bytes();
    if encoded.len() != 64 {
        return Err("plan digest has the wrong width".into());
    }
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        let offset = index * 2;
        let pair = std::str::from_utf8(&encoded[offset..offset + 2])
            .map_err(|_| "plan digest is not hexadecimal")?;
        *byte = u8::from_str_radix(pair, 16).map_err(|_| "plan digest is not hexadecimal")?;
    }
    Ok(output)
}

fn decode_silu_q7_gate(payload: &[u8]) -> Result<(SiluQ7GateHeader, &[u8]), String> {
    if payload.len() > SILU_Q7_MAX_EVALUATOR_PAYLOAD_BYTES {
        return Err("Q7 SiLU evaluator payload exceeds its byte bound".into());
    }
    let length_bytes: [u8; 4] = payload
        .get(..4)
        .ok_or("Q7 SiLU gate payload is truncated")?
        .try_into()
        .map_err(|_| "Q7 SiLU gate payload is truncated")?;
    let header_len = usize::try_from(u32::from_le_bytes(length_bytes))
        .map_err(|_| "Q7 SiLU gate header exceeds usize")?;
    let header_end = 4_usize
        .checked_add(header_len)
        .ok_or("Q7 SiLU gate header length overflows")?;
    let header_bytes = payload
        .get(4..header_end)
        .ok_or("Q7 SiLU gate header is truncated")?;
    let gate = payload
        .get(header_end..)
        .filter(|gate| !gate.is_empty())
        .ok_or("Q7 SiLU gate body is absent")?;
    let header: SiluQ7GateHeader =
        serde_json::from_slice(header_bytes).map_err(|_| "Q7 SiLU gate header is invalid")?;
    if header.schema_version != SILU_Q7_GATE_SCHEMA_VERSION
        || canonical_bytes(&header) != header_bytes
        || header.gate_digest != digest_bytes(SILU_Q7_GATE_SCHEMA_VERSION, gate)
    {
        return Err("Q7 SiLU gate commitment is invalid".into());
    }
    Ok((header, gate))
}

fn validate_silu_q7_plan(compiled: &CompiledPlan) -> Result<(usize, &RegionStep), String> {
    compiled.verify()?;
    let [step] = compiled.region_program.steps.as_slice() else {
        return Err("Q7 SiLU executor requires exactly one region step".into());
    };
    if step.operator != Operator::Silu
        || step.kernel.implementation != KernelImplementation::PllmGarbleSiluQuadraticQ7
        || step.input.numeric != NumericType::SignedFixedQ7
        || step.output.numeric != NumericType::SignedFixedQ7
        || step.input.shape != step.output.shape
        || step.input_representation != Representation::ArithmeticLabel
        || step.output_representation != Representation::ArithmeticLabel
    {
        return Err("region program is not single arithmetic-label Q7 SiLU".into());
    }
    let elements = step
        .input
        .shape
        .iter()
        .try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or("Q7 SiLU tensor element count overflows u64")
        })?;
    let elements = usize::try_from(elements).map_err(|_| "Q7 SiLU tensor exceeds usize")?;
    Ok((elements, step))
}

/// Verify lock and region program before entering `pllm-core`.
pub fn execute_wrap32(
    compiled: &CompiledPlan,
    weights: &[u8],
    input: &[u32],
    threads: usize,
    simd: bool,
) -> Result<Vec<u32>, String> {
    compiled.verify()?;
    let [step] = compiled.region_program.steps.as_slice() else {
        return Err("wrap32 executor requires exactly one region step".into());
    };
    if step.operator != Operator::Linear
        || step.kernel.implementation != KernelImplementation::PllmCoreMatrixWrap32
        || step.input.numeric != NumericType::Wrap32
        || step.output.numeric != NumericType::Wrap32
        || step.input.shape.len() != 2
        || step.output.shape.len() != 2
        || step.input.shape[0] != step.output.shape[0]
    {
        return Err("region program is not single linear wrap32".into());
    }
    let batch = usize::try_from(step.input.shape[0]).map_err(|_| "batch exceeds usize")?;
    let columns = usize::try_from(step.input.shape[1]).map_err(|_| "input width exceeds usize")?;
    let rows = usize::try_from(step.output.shape[1]).map_err(|_| "output width exceeds usize")?;
    let matrix = pllm_core::Matrix::new(weights, rows, columns)?;
    let executor = pllm_core::Executor::new(threads, simd)?;
    matrix.wrap32(&executor, input, batch)
}
