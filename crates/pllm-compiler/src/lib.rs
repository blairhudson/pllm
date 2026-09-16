//! Deterministic lowering from locked context into canonical plans plus a private region program.

use pllm_models::{DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator};
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

pub const REGION_PROGRAM_SCHEMA_VERSION: &str = "pllm.region_program.v1";
pub const COMPILE_REQUEST_SCHEMA_VERSION: &str = "pllm.compile_request.v1";
pub const BASELINE_EXPERIMENT_PROFILE: &str = "baseline.masked_linear_cpu";
pub const SILU_Q7_NUMERIC_GRAPH_ID: &str = "pllm.numeric.silu.quadratic_q7.v1";
pub const SILU_Q7_PROTECTED_GRAPH_ID: &str = "pllm.protected.arithmetic_garbling.silu_q7.v1";
pub const SILU_Q7_METHOD_ID: &str = "arithmetic-garbling-silu-q7";
pub const SILU_Q7_KERNEL_DESCRIPTOR_ID: &str = "pllm-garble-silu-quadratic-q7";
pub const SILU_Q7_COMPILER_ID: &str = "pllm-compiler";
pub const SILU_Q7_MAX_TENSOR_ELEMENTS: usize = 128;
pub const SILU_Q7_MAX_EVALUATOR_PAYLOAD_BYTES: usize = 16_384;
pub const SILU_Q7_MAX_LABEL_BYTES: usize = 1_024;
const SILU_Q7_BURN_LEDGER_CAPACITY: usize = 65_536;
const SILU_Q7_GATE_SCHEMA_VERSION: &str = "pllm.silu_q7_gate.v2";

pub fn silu_q7_kernel_artifact_digest() -> Digest {
    canonical_digest(
        "pllm.artifact.rust-source-set.v1",
        &[
            digest_bytes(
                "pllm.artifact.rust-source.v1",
                include_bytes!("../../pllm-core/src/activation.rs"),
            ),
            silu_q7_method_artifact_digest(),
        ],
    )
}

pub fn silu_q7_compiler_artifact_digest() -> Digest {
    digest_bytes("pllm.artifact.rust-source.v1", include_bytes!("lib.rs"))
}

#[derive(Serialize)]
pub struct SiluQ7InstalledContract {
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
    digest_bytes(
        "pllm.artifact.rust-source.v1",
        include_bytes!("../../pllm-garble/src/lib.rs"),
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
    let operators = occurrences
        .into_iter()
        .map(|(operator, occurrences)| {
            let executable = operator == ModelOperator::Linear;
            let primitive = matches!(
                operator,
                ModelOperator::TokenLookup
                    | ModelOperator::Reshape
                    | ModelOperator::Linear
                    | ModelOperator::RotaryEmbedding
                    | ModelOperator::KvCacheAppend
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
                } else if primitive {
                    CapabilityLevel::Primitive
                } else {
                    CapabilityLevel::Missing
                },
                component: if executable {
                    Some("pllm/compiler-wrap32@0.1.0-alpha.1".to_owned())
                } else if primitive {
                    Some("pllm/agc-project@0.1.0-alpha.1-reference".to_owned())
                } else {
                    None
                },
                blocker: if executable {
                    "single semantic linear regions execute, but whole-decoder scheduling is unavailable"
                        .to_owned()
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
    for (slot, required) in [
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
    if canonical_bytes(&document) != bytes {
        return Err(document_error(
            "compile document must use canonical compact sorted JSON bytes",
        ));
    }
    compile(&request_from_document(document))
}

pub fn compile(request: &CompileRequest) -> Result<CompiledPlan, Vec<Diagnostic>> {
    let mut diagnostics = validate_context(request);
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
    if request
        .operations
        .iter()
        .any(|operation| operation.operator == Operator::Silu)
    {
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
    Ok(BoundSiluQ7Material {
        material,
        evaluator_payload,
    })
}

static SILU_Q7_BURN_LEDGER: OnceLock<Mutex<BTreeSet<[u8; 32]>>> = OnceLock::new();

fn burn_silu_q7_material(material_ids: &BTreeSet<[u8; 32]>) -> Result<(), String> {
    let mut ledger = SILU_Q7_BURN_LEDGER
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "Q7 SiLU burn ledger is poisoned")?;
    if material_ids.iter().any(|id| ledger.contains(id)) {
        return Err("Q7 SiLU evaluator material was already bound".into());
    }
    if ledger.len().saturating_add(material_ids.len()) > SILU_Q7_BURN_LEDGER_CAPACITY {
        return Err("Q7 SiLU burn ledger capacity is exhausted".into());
    }
    ledger.extend(material_ids.iter().cloned());
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
        let mut material_ids = BTreeSet::new();
        let mut duplicate = false;
        for payload in payloads {
            let (header, gate_bytes) = decode_silu_q7_gate(payload)?;
            let gate = pllm_garble::GarbledProjection::from_bytes(gate_bytes)
                .map_err(|error| error.to_string())?;
            duplicate |= !material_ids.insert(gate.material_id());
            decoded.push((header, gate));
        }
        burn_silu_q7_material(&material_ids)?;
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

fn digest_array(digest: &Digest) -> Result<[u8; 32], String> {
    let encoded = digest.as_str().as_bytes();
    if encoded.len() != 64 {
        return Err("Q7 SiLU plan digest has the wrong width".into());
    }
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        let offset = index * 2;
        let pair = std::str::from_utf8(&encoded[offset..offset + 2])
            .map_err(|_| "Q7 SiLU plan digest is not hexadecimal")?;
        *byte =
            u8::from_str_radix(pair, 16).map_err(|_| "Q7 SiLU plan digest is not hexadecimal")?;
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
