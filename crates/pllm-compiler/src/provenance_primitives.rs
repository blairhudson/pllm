use pllm_models::{
    DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator, StateKind, StateTensor,
};
use pllm_types::{canonical_digest, digest_bytes, Digest};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;
use std::sync::OnceLock;
use zeroize::{Zeroize, Zeroizing};

pub const ROPE_Q10_REGION_SCHEMA_VERSION: &str = "pllm.rope_q10_region.v1";
pub const KV_CACHE_APPEND_Q10_REGION_SCHEMA_VERSION: &str = "pllm.kv_cache_append_q10_region.v1";
pub const KV_CACHE_VIEW_Q10_REGION_SCHEMA_VERSION: &str = "pllm.kv_cache_view_q10_region.v1";
pub const ROPE_Q10_POSITION_LAYOUT: &str = "[batch,sequence]";
pub const ROPE_Q10_RANGE_POLICY: &str = "reject_outside_signed_i16";
pub const KV_CACHE_Q10_ACTIVE_POLICY: &str = "valid_length_delta_exact_true_prefix";
pub const KV_CACHE_Q10_ZERO_PADDING_POLICY: &str = "inactive_current_rows_must_be_zero";
pub const PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS: usize = 16_777_216;
pub const PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES: usize = 1_073_741_824;
pub const PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES: usize = 1_073_741_824;
pub const PROVENANCE_PRIMITIVE_HARD_MAX_OPERATIONS: usize = 1_000_000;
pub const PROVENANCE_PRIMITIVE_HARD_MAX_STATES: usize = 1_000_000;
pub const PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_BYTES: usize = 67_108_864;
pub const PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_DEPTH: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProvenancePrimitiveResourcePolicy {
    max_rope_elements: usize,
    max_cache_bytes: usize,
    max_view_bytes: usize,
    max_operations: usize,
    max_states: usize,
    max_metadata_bytes: usize,
    max_metadata_depth: usize,
}

pub struct KvCacheQ10ExecutionInputs<'a> {
    current: &'a [i16],
    positions: &'a [u32],
    attention_mask: &'a [bool],
    valid_lengths: &'a [usize],
}

impl<'a> KvCacheQ10ExecutionInputs<'a> {
    pub const fn new(
        current: &'a [i16],
        positions: &'a [u32],
        attention_mask: &'a [bool],
        valid_lengths: &'a [usize],
    ) -> Self {
        Self {
            current,
            positions,
            attention_mask,
            valid_lengths,
        }
    }
}

impl ProvenancePrimitiveResourcePolicy {
    pub fn new(
        max_rope_elements: usize,
        max_cache_bytes: usize,
        max_view_bytes: usize,
        max_operations: usize,
        max_states: usize,
        max_metadata_bytes: usize,
        max_metadata_depth: usize,
    ) -> Result<Self, String> {
        for (name, value, hard_maximum) in [
            (
                "max_rope_elements",
                max_rope_elements,
                PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS,
            ),
            (
                "max_cache_bytes",
                max_cache_bytes,
                PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES,
            ),
            (
                "max_view_bytes",
                max_view_bytes,
                PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES,
            ),
            (
                "max_operations",
                max_operations,
                PROVENANCE_PRIMITIVE_HARD_MAX_OPERATIONS,
            ),
            (
                "max_states",
                max_states,
                PROVENANCE_PRIMITIVE_HARD_MAX_STATES,
            ),
            (
                "max_metadata_bytes",
                max_metadata_bytes,
                PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_BYTES,
            ),
            (
                "max_metadata_depth",
                max_metadata_depth,
                PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_DEPTH,
            ),
        ] {
            if value == 0 || value > hard_maximum {
                return Err(format!(
                    "{name} must be in 1..={hard_maximum}, received {value}"
                ));
            }
        }
        Ok(Self {
            max_rope_elements,
            max_cache_bytes,
            max_view_bytes,
            max_operations,
            max_states,
            max_metadata_bytes,
            max_metadata_depth,
        })
    }

    pub const fn max_rope_elements(&self) -> usize {
        self.max_rope_elements
    }

    pub const fn max_cache_bytes(&self) -> usize {
        self.max_cache_bytes
    }

    pub const fn max_view_bytes(&self) -> usize {
        self.max_view_bytes
    }

    pub const fn max_operations(&self) -> usize {
        self.max_operations
    }

    pub const fn max_states(&self) -> usize {
        self.max_states
    }

    pub const fn max_metadata_bytes(&self) -> usize {
        self.max_metadata_bytes
    }

    pub const fn max_metadata_depth(&self) -> usize {
        self.max_metadata_depth
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRopeQ10Region {
    pub schema_version: String,
    pub model_plan_digest: Digest,
    pub mode: DecoderMode,
    pub layer: Option<u64>,
    pub operation_id: String,
    pub input_id: String,
    pub positions_id: String,
    pub tensor_shape: Vec<u64>,
    pub input_layout: String,
    pub output_layout: String,
    pub positions_shape: Vec<u64>,
    pub positions_layout: String,
    pub theta: String,
    pub rotary_dimensions: u64,
    pub pairing: String,
    pub position_policy: String,
    pub coefficient_profile: String,
    pub tail_policy: String,
    pub execution_profile: String,
    pub rounding_policy: String,
    pub range_policy: String,
    pub maximum_position: u32,
    pub compiler_core_artifact_digest: Digest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheAppendMode {
    Initialize,
    Append,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelKvCacheAppendQ10Region {
    pub schema_version: String,
    pub model_plan_digest: Digest,
    pub mode: DecoderMode,
    pub operation_id: String,
    pub layer: u64,
    pub state_kind: StateKind,
    pub state_id: String,
    pub current_input_id: String,
    pub positions_id: String,
    pub attention_mask_id: String,
    pub valid_lengths_id: String,
    pub current_shape: Vec<u64>,
    pub current_layout: String,
    pub positions_shape: Vec<u64>,
    pub attention_mask_shape: Vec<u64>,
    pub valid_lengths_shape: Vec<u64>,
    pub state_shape: Vec<u64>,
    pub state_layout: String,
    pub state_capacity: u64,
    pub append_mode: KvCacheAppendMode,
    pub attention_maximum_sequence: u64,
    pub active_policy: String,
    pub zero_padding_policy: String,
    pub execution_profile: String,
    pub compiler_core_artifact_digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelKvCacheViewQ10Region {
    pub schema_version: String,
    pub model_plan_digest: Digest,
    pub mode: DecoderMode,
    pub operation_id: String,
    pub layer: u64,
    pub append_producer_id: String,
    pub state_kind: StateKind,
    pub positions_id: String,
    pub attention_mask_id: String,
    pub valid_lengths_id: String,
    pub output_shape: Vec<u64>,
    pub output_layout: String,
    pub axis: u64,
    pub maximum_sequence: u64,
    pub semantics: String,
    pub compiler_core_artifact_digest: Digest,
}

#[derive(Eq, PartialEq)]
struct KvCacheBinding {
    plan_digest: Digest,
    model_config_digest: Digest,
    model_family: String,
    adapter: String,
    layer: u64,
    state_kind: StateKind,
    state_id: String,
    state_shape: Vec<u64>,
    state_layout: String,
    state_capacity: u64,
    artifact_digest: Digest,
}

/// Opaque plan-bound cache. Contents and valid lengths are omitted from `Debug`.
pub struct ProvenanceBoundKvCacheQ10 {
    cache: pllm_core::BoundedKvCacheQ10,
    binding: KvCacheBinding,
    active_mode: DecoderMode,
    append_operation_id: String,
}

impl fmt::Debug for ProvenanceBoundKvCacheQ10 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProvenanceBoundKvCacheQ10")
            .field("layer", &self.binding.layer)
            .field("state_kind", &self.binding.state_kind)
            .field("state_id", &self.binding.state_id)
            .field("shape", &self.binding.state_shape)
            .field("contents", &"<redacted>")
            .finish()
    }
}

impl ProvenanceBoundKvCacheQ10 {
    pub fn shape(&self) -> [usize; 4] {
        self.cache.shape()
    }

    pub fn valid_lengths(&self) -> &[usize] {
        self.cache.valid_lengths()
    }
}

/// Active plaintext cache prefixes copied from bounded persistent state.
///
/// This type intentionally implements neither `Debug` nor `Clone`.
///
/// ```compile_fail
/// fn require_debug<T: std::fmt::Debug>() {}
/// require_debug::<pllm_compiler::BoundedKvCacheViewQ10>();
/// ```
///
/// ```compile_fail
/// fn require_clone<T: Clone>() {}
/// require_clone::<pllm_compiler::BoundedKvCacheViewQ10>();
/// ```
pub struct BoundedKvCacheViewQ10 {
    prefixes: Vec<Vec<Zeroizing<Vec<i16>>>>,
    valid_lengths: Zeroizing<Vec<usize>>,
    maximum_sequence: usize,
    head_dim: usize,
}

impl BoundedKvCacheViewQ10 {
    pub fn prefix(&self, batch: usize, head: usize) -> Option<&[i16]> {
        self.prefixes
            .get(batch)
            .and_then(|heads| heads.get(head))
            .map(|prefix| prefix.as_slice())
    }

    pub fn valid_lengths(&self) -> &[usize] {
        &self.valid_lengths
    }

    pub const fn maximum_sequence(&self) -> usize {
        self.maximum_sequence
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn zeroize_contents(&mut self) {
        self.prefixes.clear();
        self.valid_lengths.zeroize();
        self.maximum_sequence.zeroize();
        self.head_dim.zeroize();
    }
}

impl Drop for BoundedKvCacheViewQ10 {
    fn drop(&mut self) {
        self.zeroize_contents();
    }
}

/// Digest of complete source/configuration closure used by these reference primitives.
pub fn provenance_primitives_artifact_digest() -> &'static Digest {
    static DIGEST: OnceLock<Digest> = OnceLock::new();
    DIGEST.get_or_init(compute_provenance_primitives_artifact_digest)
}

fn compute_provenance_primitives_artifact_digest() -> Digest {
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
        "pllm.artifact.provenance_primitives.complete_source_set.v1",
        &[
            file(
                "crates/pllm-compiler/src/provenance_primitives.rs",
                include_bytes!("provenance_primitives.rs"),
            ),
            file("crates/pllm-compiler/src/lib.rs", include_bytes!("lib.rs")),
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
                "schemas/decoder-plan.schema.json",
                include_bytes!("../../../schemas/decoder-plan.schema.json"),
            ),
            file(
                "crates/pllm-types/src/lib.rs",
                include_bytes!("../../pllm-types/src/lib.rs"),
            ),
            file(
                "crates/pllm-core/src/rope.rs",
                include_bytes!("../../pllm-core/src/rope.rs"),
            ),
            file(
                "crates/pllm-core/src/kv_cache.rs",
                include_bytes!("../../pllm-core/src/kv_cache.rs"),
            ),
            file(
                "crates/pllm-core/src/fixed_point.rs",
                include_bytes!("../../pllm-core/src/fixed_point.rs"),
            ),
            file(
                "crates/pllm-core/src/lib.rs",
                include_bytes!("../../pllm-core/src/lib.rs"),
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

pub fn lower_model_rope_q10_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
    resource_policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<Vec<ModelRopeQ10Region>, String> {
    preflight_plan(plan, resource_policy)?;
    lower_model_rope_q10_regions_preflighted(plan, mode)
}

fn lower_model_rope_q10_regions_preflighted(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelRopeQ10Region>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let plan_digest = digest_admitted_plan(plan);
    let graph = graph(plan, mode);
    let count = graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::RotaryEmbedding)
        .count();
    let mut regions = Vec::new();
    regions
        .try_reserve_exact(count)
        .map_err(|_| "RoPE region output allocation failed")?;
    for operation in graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::RotaryEmbedding)
    {
        regions.push(lower_rope_region(&plan_digest, graph, mode, operation)?);
    }
    Ok(regions)
}

pub fn execute_model_rope_q10(
    plan: &DecoderPlan,
    region: &ModelRopeQ10Region,
    input: &[i16],
    positions: &[u32],
    resource_policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<Vec<i16>, String> {
    preflight_plan(plan, resource_policy)?;
    preflight_rope_region(region, resource_policy)?;
    let shape = shape4_usize(&region.tensor_shape, "RoPE tensor")?;
    let elements = shape_elements(shape, "RoPE tensor")?;
    if elements > resource_policy.max_rope_elements {
        return Err(format!(
            "RoPE requires {elements} elements, exceeding resource policy maximum {}",
            resource_policy.max_rope_elements
        ));
    }
    if input.len() != elements {
        return Err(format!(
            "RoPE requires {elements} input elements, received {}",
            input.len()
        ));
    }
    validate_sequential_absolute_positions(positions, shape, region.maximum_position)?;
    let authentic = lower_model_rope_q10_regions_preflighted(plan, region.mode)?
        .into_iter()
        .find(|candidate| candidate.operation_id == region.operation_id)
        .ok_or("RoPE region is absent from its decoder plan")?;
    if authentic != *region {
        return Err("RoPE region differs from exact decoder-plan lowering".into());
    }
    let theta = region
        .theta
        .parse::<f64>()
        .map_err(|_| "RoPE theta is not a finite positive number")?;
    let rotary_dimensions = usize::try_from(region.rotary_dimensions)
        .map_err(|_| "RoPE rotary dimensions exceed usize")?;
    pllm_core::rope_q10(
        input,
        positions,
        shape,
        pllm_core::RopeQ10Config {
            theta,
            rotary_dimensions,
            maximum_position: region.maximum_position,
        },
    )
    .map_err(|error| error.to_string())
}

pub fn lower_model_kv_cache_append_q10_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
    resource_policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<Vec<ModelKvCacheAppendQ10Region>, String> {
    preflight_plan(plan, resource_policy)?;
    lower_model_kv_cache_append_q10_regions_preflighted(plan, mode)
}

fn lower_model_kv_cache_append_q10_regions_preflighted(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelKvCacheAppendQ10Region>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let plan_digest = digest_admitted_plan(plan);
    let graph = graph(plan, mode);
    let count = graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::KvCacheAppend)
        .count();
    let mut regions = Vec::new();
    regions
        .try_reserve_exact(count)
        .map_err(|_| "KV append region output allocation failed")?;
    for operation in graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::KvCacheAppend)
    {
        regions.push(lower_append_region(
            plan,
            &plan_digest,
            graph,
            mode,
            operation,
        )?);
    }
    Ok(regions)
}

pub fn lower_model_kv_cache_view_q10_regions(
    plan: &DecoderPlan,
    mode: DecoderMode,
    resource_policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<Vec<ModelKvCacheViewQ10Region>, String> {
    preflight_plan(plan, resource_policy)?;
    lower_model_kv_cache_view_q10_regions_preflighted(plan, mode)
}

fn lower_model_kv_cache_view_q10_regions_preflighted(
    plan: &DecoderPlan,
    mode: DecoderMode,
) -> Result<Vec<ModelKvCacheViewQ10Region>, String> {
    plan.validate().map_err(|error| error.to_string())?;
    let plan_digest = digest_admitted_plan(plan);
    let graph = graph(plan, mode);
    let count = graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::CacheSuffix)
        .count();
    let mut regions = Vec::new();
    regions
        .try_reserve_exact(count)
        .map_err(|_| "KV view region output allocation failed")?;
    for operation in graph
        .operations
        .iter()
        .filter(|operation| operation.operator == ModelOperator::CacheSuffix)
    {
        regions.push(lower_view_region(
            plan,
            &plan_digest,
            graph,
            mode,
            operation,
        )?);
    }
    Ok(regions)
}

pub(crate) struct ModelProvenanceQ10Coverage {
    pub rope: Result<(), String>,
    pub cache_append: Result<(), String>,
    pub cache_view: Result<(), String>,
}

pub(crate) fn model_provenance_q10_coverage(plan: &DecoderPlan) -> ModelProvenanceQ10Coverage {
    let authenticated = preflight_plan_hard(plan).and_then(|()| {
        plan.validate()
            .map_err(|error| format!("decoder plan validation failed: {error}"))
    });
    if let Err(error) = authenticated {
        return ModelProvenanceQ10Coverage {
            rope: Err(error.clone()),
            cache_append: Err(error.clone()),
            cache_view: Err(error),
        };
    }
    let plan_digest = digest_admitted_plan(plan);
    ModelProvenanceQ10Coverage {
        rope: coverage_for_operator(plan, &plan_digest, ModelOperator::RotaryEmbedding),
        cache_append: coverage_for_operator(plan, &plan_digest, ModelOperator::KvCacheAppend),
        cache_view: coverage_for_operator(plan, &plan_digest, ModelOperator::CacheSuffix),
    }
}

fn coverage_for_operator(
    plan: &DecoderPlan,
    plan_digest: &Digest,
    operator: ModelOperator,
) -> Result<(), String> {
    let mut found = false;
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        let graph = graph(plan, mode);
        for operation in graph
            .operations
            .iter()
            .filter(|operation| operation.operator == operator)
        {
            found = true;
            let result = match operator {
                ModelOperator::RotaryEmbedding => {
                    lower_rope_region(plan_digest, graph, mode, operation).map(|_| ())
                }
                ModelOperator::KvCacheAppend => {
                    lower_append_region(plan, plan_digest, graph, mode, operation).map(|_| ())
                }
                ModelOperator::CacheSuffix => {
                    lower_view_region(plan, plan_digest, graph, mode, operation).map(|_| ())
                }
                _ => Err("operator has no provenance-bound Q10 coverage check".into()),
            };
            result.map_err(|error| format!("operation {}: {error}", operation.id))?;
        }
    }
    if !found {
        return Err(format!("decoder plan has no {operator:?} operations"));
    }
    Ok(())
}

pub fn initialize_model_kv_cache_q10(
    plan: &DecoderPlan,
    region: &ModelKvCacheAppendQ10Region,
    inputs: KvCacheQ10ExecutionInputs<'_>,
    resource_policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<ProvenanceBoundKvCacheQ10, String> {
    preflight_plan(plan, resource_policy)?;
    preflight_append_region(region, resource_policy)?;
    let required_bytes = cache_required_bytes(&region.state_shape)?;
    if required_bytes > resource_policy.max_cache_bytes {
        return Err(format!(
            "KV cache requires {required_bytes} bytes, exceeding resource policy maximum {}",
            resource_policy.max_cache_bytes
        ));
    }
    validate_authentic_append_region_preflighted(plan, region)?;
    if region.mode != DecoderMode::Prefill || region.append_mode != KvCacheAppendMode::Initialize {
        return Err("KV cache initialization requires prefill initialize region".into());
    }
    let binding = binding(plan, region)?;
    let append_operation_id =
        copy_region_string(&region.operation_id, "KV cache active append operation ID")?;
    let [batch, heads, query, head_dim] = shape4_usize(&region.current_shape, "current KV")?;
    let capacity =
        usize::try_from(region.state_capacity).map_err(|_| "KV cache capacity exceeds usize")?;
    let cache = pllm_core::BoundedKvCacheQ10::initialize(
        batch,
        heads,
        capacity,
        head_dim,
        inputs.current,
        query,
        inputs.positions,
        inputs.attention_mask,
        inputs.valid_lengths,
    )
    .map_err(|error| error.to_string())?;
    Ok(ProvenanceBoundKvCacheQ10 {
        cache,
        binding,
        active_mode: region.mode,
        append_operation_id,
    })
}

pub fn append_model_kv_cache_q10(
    cache: &mut ProvenanceBoundKvCacheQ10,
    plan: &DecoderPlan,
    region: &ModelKvCacheAppendQ10Region,
    inputs: KvCacheQ10ExecutionInputs<'_>,
    resource_policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<(), String> {
    preflight_plan(plan, resource_policy)?;
    preflight_append_region(region, resource_policy)?;
    let required_bytes = cache_required_bytes(&region.state_shape)?;
    if required_bytes > resource_policy.max_cache_bytes {
        return Err(format!(
            "KV cache requires {required_bytes} bytes, exceeding resource policy maximum {}",
            resource_policy.max_cache_bytes
        ));
    }
    validate_authentic_append_region_preflighted(plan, region)?;
    if region.mode != DecoderMode::Decode || region.append_mode != KvCacheAppendMode::Append {
        return Err("KV cache append requires decode append region".into());
    }
    validate_continuation(&cache.binding, plan, region)?;
    let next_append_operation_id =
        copy_region_string(&region.operation_id, "KV cache active append operation ID")?;
    let query = usize::try_from(
        *region
            .current_shape
            .get(2)
            .ok_or("current KV shape is not rank four")?,
    )
    .map_err(|_| "KV cache query extent exceeds usize")?;
    cache
        .cache
        .append(
            inputs.current,
            query,
            inputs.positions,
            inputs.attention_mask,
            inputs.valid_lengths,
        )
        .map_err(|error| error.to_string())?;
    cache.active_mode = region.mode;
    cache.append_operation_id = next_append_operation_id;
    Ok(())
}

pub fn execute_model_kv_cache_view_q10(
    cache: &ProvenanceBoundKvCacheQ10,
    plan: &DecoderPlan,
    region: &ModelKvCacheViewQ10Region,
    resource_policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<BoundedKvCacheViewQ10, String> {
    preflight_plan(plan, resource_policy)?;
    preflight_view_region(region, resource_policy)?;
    let required_bytes = view_required_bytes(&region.output_shape)?;
    if required_bytes > resource_policy.max_view_bytes {
        return Err(format!(
            "KV cache view requires {required_bytes} bytes, exceeding resource policy maximum {}",
            resource_policy.max_view_bytes
        ));
    }
    let authentic = lower_model_kv_cache_view_q10_regions_preflighted(plan, region.mode)?
        .into_iter()
        .find(|candidate| candidate.operation_id == region.operation_id)
        .ok_or("KV cache view region is absent from its decoder plan")?;
    if authentic != *region {
        return Err("KV cache view region differs from exact decoder-plan lowering".into());
    }
    if cache.binding.plan_digest != region.model_plan_digest
        || cache.binding.model_config_digest != plan.config_digest
        || cache.binding.model_family != plan.model_family
        || cache.binding.adapter != plan.adapter
        || cache.binding.layer != region.layer
        || cache.binding.state_kind != region.state_kind
        || cache.active_mode != region.mode
        || cache.append_operation_id != region.append_producer_id
        || cache.binding.artifact_digest != region.compiler_core_artifact_digest
    {
        return Err("KV cache view is incompatible with bound cache state".into());
    }

    let maximum_sequence = usize::try_from(region.maximum_sequence)
        .map_err(|_| "KV cache view maximum sequence exceeds usize")?;
    let [batch, heads, _, head_dim] = cache.cache.shape();
    copy_bounded_cache_view(
        &cache.cache,
        batch,
        heads,
        maximum_sequence,
        head_dim,
        |_| Ok(()),
    )
}

fn copy_bounded_cache_view(
    cache: &pllm_core::BoundedKvCacheQ10,
    batch: usize,
    heads: usize,
    maximum_sequence: usize,
    head_dim: usize,
    mut before_prefix_copy: impl FnMut(usize) -> Result<(), String>,
) -> Result<BoundedKvCacheViewQ10, String> {
    let mut prefixes = Vec::new();
    prefixes
        .try_reserve_exact(batch)
        .map_err(|_| "KV cache view allocation failed")?;
    let mut copied_prefixes = 0usize;
    for batch_index in 0..batch {
        let mut batch_prefixes = Vec::new();
        batch_prefixes
            .try_reserve_exact(heads)
            .map_err(|_| "KV cache view allocation failed")?;
        for head_index in 0..heads {
            before_prefix_copy(copied_prefixes)?;
            let source = cache
                .visible_prefix(batch_index, head_index, maximum_sequence)
                .map_err(|error| error.to_string())?;
            let mut prefix = Zeroizing::new(Vec::new());
            prefix
                .try_reserve_exact(source.len())
                .map_err(|_| "KV cache view allocation failed")?;
            prefix.extend_from_slice(source);
            batch_prefixes.push(prefix);
            copied_prefixes = copied_prefixes
                .checked_add(1)
                .ok_or_else(|| "KV cache view prefix count overflowed".to_owned())?;
        }
        prefixes.push(batch_prefixes);
    }
    let mut valid_lengths = Zeroizing::new(Vec::new());
    valid_lengths
        .try_reserve_exact(batch)
        .map_err(|_| "KV cache view allocation failed")?;
    valid_lengths.extend_from_slice(cache.valid_lengths());
    Ok(BoundedKvCacheViewQ10 {
        prefixes,
        valid_lengths,
        maximum_sequence,
        head_dim,
    })
}

fn lower_rope_region(
    plan_digest: &Digest,
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelRopeQ10Region, String> {
    let [input_id, positions_id] = operation.inputs.as_slice() else {
        return Err(format!(
            "semantic RoPE operation {} must have exactly two inputs",
            operation.id
        ));
    };
    let input = operation_source(graph, input_id, &operation.id, "RoPE")?;
    if input.output_shape != operation.output_shape
        || operation.output_shape.len() != 4
        || operation.output_shape.contains(&0)
        || operation.output_shape[0] != graph.batch
        || operation.output_shape[2] != graph.query_sequence
    {
        return Err(format!(
            "semantic RoPE operation {} has invalid workload shape",
            operation.id
        ));
    }
    let rope_shape = shape4_usize(&operation.output_shape, "RoPE tensor")?;
    let rope_elements = shape_elements(rope_shape, "RoPE tensor")?;
    if rope_elements > PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS {
        return Err(format!(
            "semantic RoPE operation {} requires {rope_elements} elements, exceeding immutable hard maximum {}",
            operation.id, PROVENANCE_PRIMITIVE_HARD_MAX_ROPE_ELEMENTS
        ));
    }
    let attributes = exact_attributes(
        operation,
        &[
            "theta",
            "rotary_dimensions",
            "pairing",
            "position_policy",
            "coefficient_profile",
            "input_layout",
            "output_layout",
            "tail_policy",
        ],
    )?;
    let theta = attributes
        .get("theta")
        .and_then(Value::as_number)
        .ok_or_else(|| format!("semantic RoPE operation {} has invalid theta", operation.id))?;
    let theta = copy_json_number(theta, "RoPE theta")?;
    let parsed_theta = theta
        .parse::<f64>()
        .map_err(|_| format!("semantic RoPE operation {} has invalid theta", operation.id))?;
    if !parsed_theta.is_finite() || parsed_theta <= 0.0 {
        return Err(format!(
            "semantic RoPE operation {} has invalid theta",
            operation.id
        ));
    }
    let rotary_dimensions = u64_attribute(attributes, "rotary_dimensions", operation)?;
    let head_dim = operation.output_shape[3];
    if rotary_dimensions == 0 || rotary_dimensions % 2 != 0 || rotary_dimensions > head_dim {
        return Err(format!(
            "semantic RoPE operation {} has invalid rotary dimensions",
            operation.id
        ));
    }
    let pairing = exact_string(attributes, "pairing", "split_half", operation)?;
    let position_policy = exact_string(
        attributes,
        "position_policy",
        "sequential_absolute",
        operation,
    )?;
    let coefficient_profile = exact_string(
        attributes,
        "coefficient_profile",
        pllm_core::ROPE_Q10_COEFFICIENT_PROFILE,
        operation,
    )?;
    let input_layout = exact_string(
        attributes,
        "input_layout",
        "batch_heads_sequence_feature",
        operation,
    )?;
    let output_layout = exact_string(
        attributes,
        "output_layout",
        "batch_heads_sequence_feature",
        operation,
    )?;
    let tail_policy = exact_string(attributes, "tail_policy", "unchanged", operation)?;
    if positions_id.is_empty() {
        return Err(format!(
            "semantic RoPE operation {} has no positions producer identity",
            operation.id
        ));
    }
    let maximum_position = maximum_position(graph, operation.layer)?;
    Ok(ModelRopeQ10Region {
        schema_version: copy_region_string(ROPE_Q10_REGION_SCHEMA_VERSION, "RoPE schema version")?,
        model_plan_digest: copy_region_digest(plan_digest, "RoPE plan digest")?,
        mode,
        layer: operation.layer,
        operation_id: copy_region_string(&operation.id, "RoPE operation ID")?,
        input_id: copy_region_string(input_id, "RoPE input ID")?,
        positions_id: copy_region_string(positions_id, "RoPE positions ID")?,
        tensor_shape: copy_region_shape(&operation.output_shape, "RoPE tensor shape")?,
        input_layout: copy_region_string(input_layout, "RoPE input layout")?,
        output_layout: copy_region_string(output_layout, "RoPE output layout")?,
        positions_shape: copy_region_shape(
            &[graph.batch, graph.query_sequence],
            "RoPE positions shape",
        )?,
        positions_layout: copy_region_string(ROPE_Q10_POSITION_LAYOUT, "RoPE positions layout")?,
        theta,
        rotary_dimensions,
        pairing: copy_region_string(pairing, "RoPE pairing")?,
        position_policy: copy_region_string(position_policy, "RoPE position policy")?,
        coefficient_profile: copy_region_string(coefficient_profile, "RoPE coefficient profile")?,
        tail_policy: copy_region_string(tail_policy, "RoPE tail policy")?,
        execution_profile: copy_region_string(
            pllm_core::ROPE_Q10_PROFILE,
            "RoPE execution profile",
        )?,
        rounding_policy: copy_region_string(
            pllm_core::rope::ROPE_Q10_ROUNDING,
            "RoPE rounding policy",
        )?,
        range_policy: copy_region_string(ROPE_Q10_RANGE_POLICY, "RoPE range policy")?,
        maximum_position,
        compiler_core_artifact_digest: copy_region_digest(
            provenance_primitives_artifact_digest(),
            "RoPE artifact digest",
        )?,
    })
}

fn lower_append_region(
    plan: &DecoderPlan,
    plan_digest: &Digest,
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelKvCacheAppendQ10Region, String> {
    let layer = operation
        .layer
        .ok_or_else(|| format!("KV cache append {} has no layer", operation.id))?;
    let state_kind = operation
        .state_kind
        .filter(|kind| matches!(kind, StateKind::Key | StateKind::Value))
        .ok_or_else(|| format!("KV cache append {} has invalid state kind", operation.id))?;
    let attributes = exact_attributes(
        operation,
        &[
            "state",
            "mode",
            "state_capacity",
            "state_layout",
            "absolute_write_positions_input",
            "padding_mask_input",
            "valid_lengths_input",
            "attention_domain",
        ],
    )?;
    let append_mode = match (mode, attributes.get("mode").and_then(Value::as_str)) {
        (DecoderMode::Prefill, Some("initialize")) => KvCacheAppendMode::Initialize,
        (DecoderMode::Decode, Some("append")) => KvCacheAppendMode::Append,
        _ => {
            return Err(format!(
                "KV cache append {} mode differs from decoder graph",
                operation.id
            ))
        }
    };
    let expected_arity = if append_mode == KvCacheAppendMode::Initialize {
        4
    } else {
        5
    };
    if operation.inputs.len() != expected_arity {
        return Err(format!(
            "KV cache append {} has invalid arity",
            operation.id
        ));
    }
    let offset = usize::from(append_mode == KvCacheAppendMode::Append);
    let current_input_id = &operation.inputs[offset];
    let positions_id = &operation.inputs[offset + 1];
    let attention_mask_id = &operation.inputs[offset + 2];
    let valid_lengths_id = &operation.inputs[offset + 3];
    let current = operation_source(graph, current_input_id, &operation.id, "KV cache append")?;
    if current.output_shape.len() != 4
        || current.output_shape.contains(&0)
        || current.output_shape[0] != graph.batch
        || current.output_shape[2] != graph.query_sequence
    {
        return Err(format!(
            "KV cache append {} current source has invalid workload shape",
            operation.id
        ));
    }
    shape4_usize(&current.output_shape, "current KV")?;
    let state_id = nonempty_string_attribute(attributes, "state", operation)?;
    if append_mode == KvCacheAppendMode::Append
        && operation.inputs.first().map(String::as_str) != Some(state_id)
    {
        return Err(format!(
            "KV cache append {} does not consume declared state slot",
            operation.id
        ));
    }
    require_attribute_input(
        attributes,
        "absolute_write_positions_input",
        positions_id,
        operation,
    )?;
    require_attribute_input(
        attributes,
        "padding_mask_input",
        attention_mask_id,
        operation,
    )?;
    require_attribute_input(
        attributes,
        "valid_lengths_input",
        valid_lengths_id,
        operation,
    )?;
    let state_layout = exact_string(
        attributes,
        "state_layout",
        "batch_kv_heads_sequence_feature",
        operation,
    )?;
    let state_capacity = u64_attribute(attributes, "state_capacity", operation)?;
    if state_capacity == 0 {
        return Err(format!(
            "KV cache append {} has zero capacity",
            operation.id
        ));
    }
    let expected_state_shape = [
        current.output_shape[0],
        current.output_shape[1],
        state_capacity,
        current.output_shape[3],
    ];
    if operation.output_shape.as_slice() != expected_state_shape {
        return Err(format!(
            "KV cache append {} persistent output shape is invalid",
            operation.id
        ));
    }
    shape4_usize(&operation.output_shape, "persistent KV state")?;
    let cache_bytes = cache_required_bytes(&operation.output_shape)?;
    if cache_bytes > PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES {
        return Err(format!(
            "KV cache append {} requires {cache_bytes} bytes, exceeding immutable hard maximum {}",
            operation.id, PROVENANCE_PRIMITIVE_HARD_MAX_CACHE_BYTES
        ));
    }
    let output_state = exact_state(
        &graph.state_outputs,
        &operation.id,
        layer,
        state_kind,
        &operation.id,
    )?;
    validate_state_descriptor(
        output_state,
        &operation.output_shape,
        state_capacity,
        operation,
    )?;
    let decode_state =
        exact_state_slot(&plan.decode.state_inputs, layer, state_kind, &operation.id)?;
    if decode_state.id != state_id {
        return Err(format!(
            "KV cache append {} state identity differs from decode state slot",
            operation.id
        ));
    }
    validate_state_descriptor(
        decode_state,
        &operation.output_shape,
        state_capacity,
        operation,
    )?;
    if append_mode == KvCacheAppendMode::Append {
        let graph_state = exact_state(
            &graph.state_inputs,
            state_id,
            layer,
            state_kind,
            &operation.id,
        )?;
        validate_state_descriptor(
            graph_state,
            &operation.output_shape,
            state_capacity,
            operation,
        )?;
    }
    let attention_domain = attributes
        .get("attention_domain")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            format!(
                "KV cache append {} has invalid attention domain",
                operation.id
            )
        })?;
    if attention_domain.len() != 3
        || !attention_domain.contains_key("layout")
        || !attention_domain.contains_key("maximum_sequence")
        || !attention_domain.contains_key("includes_current")
        || attention_domain.get("layout").and_then(Value::as_str)
            != Some("batch_kv_heads_sequence_feature")
        || attention_domain
            .get("includes_current")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(format!(
            "KV cache append {} has unsupported attention domain",
            operation.id
        ));
    }
    let attention_maximum_sequence = attention_domain
        .get("maximum_sequence")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("KV cache append {} has invalid visible bound", operation.id))?;
    if attention_maximum_sequence == 0
        || attention_maximum_sequence > state_capacity
        || attention_maximum_sequence != graph.maximum_key_sequence
    {
        return Err(format!(
            "KV cache append {} has invalid visible bound",
            operation.id
        ));
    }
    Ok(ModelKvCacheAppendQ10Region {
        schema_version: copy_region_string(
            KV_CACHE_APPEND_Q10_REGION_SCHEMA_VERSION,
            "KV append schema version",
        )?,
        model_plan_digest: copy_region_digest(plan_digest, "KV append plan digest")?,
        mode,
        operation_id: copy_region_string(&operation.id, "KV append operation ID")?,
        layer,
        state_kind,
        state_id: copy_region_string(state_id, "KV append state ID")?,
        current_input_id: copy_region_string(current_input_id, "KV append current input ID")?,
        positions_id: copy_region_string(positions_id, "KV append positions ID")?,
        attention_mask_id: copy_region_string(attention_mask_id, "KV append attention mask ID")?,
        valid_lengths_id: copy_region_string(valid_lengths_id, "KV append valid lengths ID")?,
        current_shape: copy_region_shape(&current.output_shape, "KV append current shape")?,
        current_layout: copy_region_string(
            pllm_core::kv_cache::Q10_KV_CURRENT_LAYOUT,
            "KV append current layout",
        )?,
        positions_shape: copy_region_shape(
            &[graph.batch, graph.query_sequence],
            "KV append positions shape",
        )?,
        attention_mask_shape: copy_region_shape(
            &[graph.batch, graph.query_sequence],
            "KV append attention mask shape",
        )?,
        valid_lengths_shape: copy_region_shape(&[graph.batch], "KV append valid lengths shape")?,
        state_shape: copy_region_shape(&operation.output_shape, "KV append state shape")?,
        state_layout: copy_region_string(state_layout, "KV append state layout")?,
        state_capacity,
        append_mode,
        attention_maximum_sequence,
        active_policy: copy_region_string(KV_CACHE_Q10_ACTIVE_POLICY, "KV append active policy")?,
        zero_padding_policy: copy_region_string(
            KV_CACHE_Q10_ZERO_PADDING_POLICY,
            "KV append zero-padding policy",
        )?,
        execution_profile: copy_region_string(
            pllm_core::kv_cache::Q10_KV_CACHE_PROFILE,
            "KV append execution profile",
        )?,
        compiler_core_artifact_digest: copy_region_digest(
            provenance_primitives_artifact_digest(),
            "KV append artifact digest",
        )?,
    })
}

fn lower_view_region(
    plan: &DecoderPlan,
    plan_digest: &Digest,
    graph: &DecoderGraph,
    mode: DecoderMode,
    operation: &ModelOperation,
) -> Result<ModelKvCacheViewQ10Region, String> {
    let layer = operation
        .layer
        .ok_or_else(|| format!("KV cache view {} has no layer", operation.id))?;
    let state_kind = operation
        .state_kind
        .filter(|kind| matches!(kind, StateKind::Key | StateKind::Value))
        .ok_or_else(|| format!("KV cache view {} has invalid state kind", operation.id))?;
    let [append_producer_id, positions_id, attention_mask_id, valid_lengths_id] =
        operation.inputs.as_slice()
    else {
        return Err(format!("KV cache view {} has invalid arity", operation.id));
    };
    let append = graph
        .operations
        .iter()
        .find(|candidate| candidate.id == *append_producer_id)
        .ok_or_else(|| format!("KV cache view {} has missing append producer", operation.id))?;
    if append.operator != ModelOperator::KvCacheAppend
        || append.layer != Some(layer)
        || append.state_kind != Some(state_kind)
    {
        return Err(format!(
            "KV cache view {} does not consume matching append producer",
            operation.id
        ));
    }
    let append_region = lower_append_region(plan, plan_digest, graph, mode, append)?;
    if positions_id != &append_region.positions_id
        || attention_mask_id != &append_region.attention_mask_id
        || valid_lengths_id != &append_region.valid_lengths_id
    {
        return Err(format!(
            "KV cache view {} input provenance differs from append",
            operation.id
        ));
    }
    let attributes = exact_attributes(
        operation,
        &[
            "axis",
            "maximum_sequence",
            "semantics",
            "absolute_write_positions_input",
            "padding_mask_input",
            "valid_lengths_input",
        ],
    )?;
    require_attribute_input(
        attributes,
        "absolute_write_positions_input",
        positions_id,
        operation,
    )?;
    require_attribute_input(
        attributes,
        "padding_mask_input",
        attention_mask_id,
        operation,
    )?;
    require_attribute_input(
        attributes,
        "valid_lengths_input",
        valid_lengths_id,
        operation,
    )?;
    let axis = u64_attribute(attributes, "axis", operation)?;
    let maximum_sequence = u64_attribute(attributes, "maximum_sequence", operation)?;
    let semantics = exact_string(attributes, "semantics", "visible_valid_prefix", operation)?;
    let expected_shape = [
        append_region.state_shape[0],
        append_region.state_shape[1],
        maximum_sequence,
        append_region.state_shape[3],
    ];
    if axis != 2
        || maximum_sequence == 0
        || maximum_sequence > append_region.state_capacity
        || maximum_sequence != append_region.attention_maximum_sequence
        || operation.output_shape.as_slice() != expected_shape
    {
        return Err(format!(
            "KV cache view {} has invalid bounded output shape",
            operation.id
        ));
    }
    shape4_usize(&operation.output_shape, "KV cache view")?;
    let view_bytes = view_required_bytes(&operation.output_shape)?;
    if view_bytes > PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES {
        return Err(format!(
            "KV cache view {} requires {view_bytes} bytes, exceeding immutable hard maximum {}",
            operation.id, PROVENANCE_PRIMITIVE_HARD_MAX_VIEW_BYTES
        ));
    }
    Ok(ModelKvCacheViewQ10Region {
        schema_version: copy_region_string(
            KV_CACHE_VIEW_Q10_REGION_SCHEMA_VERSION,
            "KV view schema version",
        )?,
        model_plan_digest: copy_region_digest(plan_digest, "KV view plan digest")?,
        mode,
        operation_id: copy_region_string(&operation.id, "KV view operation ID")?,
        layer,
        append_producer_id: copy_region_string(append_producer_id, "KV view append producer ID")?,
        state_kind,
        positions_id: copy_region_string(positions_id, "KV view positions ID")?,
        attention_mask_id: copy_region_string(attention_mask_id, "KV view attention mask ID")?,
        valid_lengths_id: copy_region_string(valid_lengths_id, "KV view valid lengths ID")?,
        output_shape: copy_region_shape(&operation.output_shape, "KV view output shape")?,
        output_layout: copy_region_string(
            pllm_core::kv_cache::Q10_KV_CACHE_LAYOUT,
            "KV view output layout",
        )?,
        axis,
        maximum_sequence,
        semantics: copy_region_string(semantics, "KV view semantics")?,
        compiler_core_artifact_digest: copy_region_digest(
            provenance_primitives_artifact_digest(),
            "KV view artifact digest",
        )?,
    })
}

fn validate_authentic_append_region_preflighted(
    plan: &DecoderPlan,
    region: &ModelKvCacheAppendQ10Region,
) -> Result<(), String> {
    let authentic = lower_model_kv_cache_append_q10_regions_preflighted(plan, region.mode)?
        .into_iter()
        .find(|candidate| candidate.operation_id == region.operation_id)
        .ok_or("KV cache append region is absent from its decoder plan")?;
    if authentic != *region {
        return Err("KV cache append region differs from exact decoder-plan lowering".into());
    }
    Ok(())
}

fn binding(
    plan: &DecoderPlan,
    region: &ModelKvCacheAppendQ10Region,
) -> Result<KvCacheBinding, String> {
    Ok(KvCacheBinding {
        plan_digest: copy_region_digest(&region.model_plan_digest, "KV binding plan digest")?,
        model_config_digest: copy_region_digest(
            &plan.config_digest,
            "KV binding model config digest",
        )?,
        model_family: copy_region_string(&plan.model_family, "KV binding model family")?,
        adapter: copy_region_string(&plan.adapter, "KV binding adapter")?,
        layer: region.layer,
        state_kind: region.state_kind,
        state_id: copy_region_string(&region.state_id, "KV binding state ID")?,
        state_shape: copy_region_shape(&region.state_shape, "KV binding state shape")?,
        state_layout: copy_region_string(&region.state_layout, "KV binding state layout")?,
        state_capacity: region.state_capacity,
        artifact_digest: copy_region_digest(
            &region.compiler_core_artifact_digest,
            "KV binding artifact digest",
        )?,
    })
}

fn validate_continuation(
    binding: &KvCacheBinding,
    plan: &DecoderPlan,
    region: &ModelKvCacheAppendQ10Region,
) -> Result<(), String> {
    if binding.plan_digest != region.model_plan_digest
        || binding.model_config_digest != plan.config_digest
        || binding.model_family != plan.model_family
        || binding.adapter != plan.adapter
        || binding.layer != region.layer
        || binding.state_kind != region.state_kind
        || binding.state_id != region.state_id
        || binding.state_shape != region.state_shape
        || binding.state_layout != region.state_layout
        || binding.state_capacity != region.state_capacity
        || binding.artifact_digest != region.compiler_core_artifact_digest
    {
        return Err("decode KV append is not a compatible continuation of prefill state".into());
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PlanAuthenticationLimits {
    max_operations: usize,
    max_states: usize,
    max_metadata_bytes: usize,
    max_metadata_depth: usize,
}

impl PlanAuthenticationLimits {
    const HARD: Self = Self {
        max_operations: PROVENANCE_PRIMITIVE_HARD_MAX_OPERATIONS,
        max_states: PROVENANCE_PRIMITIVE_HARD_MAX_STATES,
        max_metadata_bytes: PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_BYTES,
        max_metadata_depth: PROVENANCE_PRIMITIVE_HARD_MAX_METADATA_DEPTH,
    };

    const fn from_policy(policy: &ProvenancePrimitiveResourcePolicy) -> Self {
        Self {
            max_operations: policy.max_operations,
            max_states: policy.max_states,
            max_metadata_bytes: policy.max_metadata_bytes,
            max_metadata_depth: policy.max_metadata_depth,
        }
    }
}

struct MetadataBudget {
    bytes: usize,
    limits: PlanAuthenticationLimits,
}

impl MetadataBudget {
    fn new(limits: PlanAuthenticationLimits) -> Self {
        Self { bytes: 0, limits }
    }

    fn add(&mut self, bytes: usize) -> Result<(), String> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or("plan metadata size overflow")?;
        if self.bytes > self.limits.max_metadata_bytes {
            return Err(format!(
                "plan metadata requires more than {} bytes",
                self.limits.max_metadata_bytes
            ));
        }
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), String> {
        self.add(value.len())
    }

    fn shape(&mut self, value: &[u64]) -> Result<(), String> {
        self.add(
            value
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or("plan metadata size overflow")?,
        )
    }
}

fn preflight_plan(
    plan: &DecoderPlan,
    policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<(), String> {
    preflight_plan_with_limits(plan, PlanAuthenticationLimits::from_policy(policy))
}

fn preflight_plan_hard(plan: &DecoderPlan) -> Result<(), String> {
    preflight_plan_with_limits(plan, PlanAuthenticationLimits::HARD)
        .map_err(|error| format!("decoder plan exceeds immutable hard limits: {error}"))
}

fn preflight_plan_with_limits(
    plan: &DecoderPlan,
    limits: PlanAuthenticationLimits,
) -> Result<(), String> {
    let operations = plan
        .prefill
        .operations
        .len()
        .checked_add(plan.decode.operations.len())
        .ok_or("decoder plan operation count overflow")?;
    if operations > limits.max_operations {
        return Err(format!(
            "decoder plan has {operations} operations, exceeding maximum {}",
            limits.max_operations
        ));
    }
    let states = plan
        .prefill
        .state_inputs
        .len()
        .checked_add(plan.prefill.state_outputs.len())
        .and_then(|count| count.checked_add(plan.decode.state_inputs.len()))
        .and_then(|count| count.checked_add(plan.decode.state_outputs.len()))
        .ok_or("decoder plan state count overflow")?;
    if states > limits.max_states {
        return Err(format!(
            "decoder plan has {states} states, exceeding maximum {}",
            limits.max_states
        ));
    }

    let mut budget = MetadataBudget::new(limits);
    budget.add(std::mem::size_of::<DecoderPlan>())?;
    budget.string(&plan.schema_version)?;
    budget.string(&plan.model_family)?;
    budget.string(&plan.adapter)?;
    budget.string(plan.config_digest.as_str())?;
    for transformation in &plan.transformations {
        budget.add(std::mem::size_of_val(transformation))?;
        budget.string(&transformation.component)?;
        budget.string(&transformation.implementation)?;
        budget.string(&transformation.method_id)?;
        budget.string(transformation.input_digest.as_str())?;
        budget.string(transformation.configuration_digest.as_str())?;
    }
    for graph in [&plan.prefill, &plan.decode] {
        account_graph_metadata(graph, &mut budget)?;
    }
    Ok(())
}

fn account_graph_metadata(graph: &DecoderGraph, budget: &mut MetadataBudget) -> Result<(), String> {
    budget.add(std::mem::size_of::<DecoderGraph>())?;
    budget.string(&graph.output)?;
    for operation in &graph.operations {
        budget.add(std::mem::size_of::<ModelOperation>())?;
        budget.string(&operation.id)?;
        for input in &operation.inputs {
            budget.string(input)?;
        }
        budget.shape(&operation.output_shape)?;
        account_json_value(&operation.attributes, budget)?;
    }
    for state in graph.state_inputs.iter().chain(&graph.state_outputs) {
        budget.add(std::mem::size_of_val(state))?;
        budget.string(&state.id)?;
        budget.shape(&state.shape)?;
    }
    Ok(())
}

fn account_json_value(value: &Value, budget: &mut MetadataBudget) -> Result<(), String> {
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(1)
        .map_err(|_| "plan metadata traversal allocation failed")?;
    pending.push((value, 1usize));
    while let Some((value, depth)) = pending.pop() {
        if depth > budget.limits.max_metadata_depth {
            return Err(format!(
                "plan metadata depth {depth} exceeds maximum {}",
                budget.limits.max_metadata_depth
            ));
        }
        match value {
            Value::Null | Value::Bool(_) => budget.add(1)?,
            Value::Number(_) => budget.add(32)?,
            Value::String(value) => budget.string(value)?,
            Value::Array(values) => {
                let child_depth = depth.checked_add(1).ok_or("plan metadata depth overflow")?;
                if !values.is_empty() && child_depth > budget.limits.max_metadata_depth {
                    return Err(format!(
                        "plan metadata depth {child_depth} exceeds maximum {}",
                        budget.limits.max_metadata_depth
                    ));
                }
                budget.add(
                    values
                        .len()
                        .checked_mul(std::mem::size_of::<Value>())
                        .ok_or("plan metadata size overflow")?,
                )?;
                pending
                    .try_reserve(values.len())
                    .map_err(|_| "plan metadata traversal allocation failed")?;
                for value in values {
                    pending.push((value, child_depth));
                }
            }
            Value::Object(values) => {
                let child_depth = depth.checked_add(1).ok_or("plan metadata depth overflow")?;
                if !values.is_empty() && child_depth > budget.limits.max_metadata_depth {
                    return Err(format!(
                        "plan metadata depth {child_depth} exceeds maximum {}",
                        budget.limits.max_metadata_depth
                    ));
                }
                budget.add(
                    values
                        .len()
                        .checked_mul(std::mem::size_of::<(String, Value)>())
                        .ok_or("plan metadata size overflow")?,
                )?;
                pending
                    .try_reserve(values.len())
                    .map_err(|_| "plan metadata traversal allocation failed")?;
                for (key, value) in values {
                    budget.string(key)?;
                    pending.push((value, child_depth));
                }
            }
        }
    }
    Ok(())
}

fn preflight_rope_region(
    region: &ModelRopeQ10Region,
    policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<(), String> {
    preflight_flat_region(
        std::mem::size_of::<ModelRopeQ10Region>(),
        &[
            &region.schema_version,
            region.model_plan_digest.as_str(),
            &region.operation_id,
            &region.input_id,
            &region.positions_id,
            &region.input_layout,
            &region.output_layout,
            &region.positions_layout,
            &region.theta,
            &region.pairing,
            &region.position_policy,
            &region.coefficient_profile,
            &region.tail_policy,
            &region.execution_profile,
            &region.rounding_policy,
            &region.range_policy,
            region.compiler_core_artifact_digest.as_str(),
        ],
        &[&region.tensor_shape, &region.positions_shape],
        policy,
    )
}

fn preflight_append_region(
    region: &ModelKvCacheAppendQ10Region,
    policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<(), String> {
    preflight_flat_region(
        std::mem::size_of::<ModelKvCacheAppendQ10Region>(),
        &[
            &region.schema_version,
            region.model_plan_digest.as_str(),
            &region.operation_id,
            &region.state_id,
            &region.current_input_id,
            &region.positions_id,
            &region.attention_mask_id,
            &region.valid_lengths_id,
            &region.current_layout,
            &region.state_layout,
            &region.active_policy,
            &region.zero_padding_policy,
            &region.execution_profile,
            region.compiler_core_artifact_digest.as_str(),
        ],
        &[
            &region.current_shape,
            &region.positions_shape,
            &region.attention_mask_shape,
            &region.valid_lengths_shape,
            &region.state_shape,
        ],
        policy,
    )
}

fn preflight_view_region(
    region: &ModelKvCacheViewQ10Region,
    policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<(), String> {
    preflight_flat_region(
        std::mem::size_of::<ModelKvCacheViewQ10Region>(),
        &[
            &region.schema_version,
            region.model_plan_digest.as_str(),
            &region.operation_id,
            &region.append_producer_id,
            &region.positions_id,
            &region.attention_mask_id,
            &region.valid_lengths_id,
            &region.output_layout,
            &region.semantics,
            region.compiler_core_artifact_digest.as_str(),
        ],
        &[&region.output_shape],
        policy,
    )
}

fn preflight_flat_region(
    fixed_bytes: usize,
    strings: &[&str],
    shapes: &[&[u64]],
    policy: &ProvenancePrimitiveResourcePolicy,
) -> Result<(), String> {
    if policy.max_metadata_depth < 1 {
        return Err("region metadata depth exceeds maximum 0".into());
    }
    let mut budget = MetadataBudget::new(PlanAuthenticationLimits::from_policy(policy));
    budget.add(fixed_bytes)?;
    for value in strings {
        budget.string(value)?;
    }
    for shape in shapes {
        budget.shape(shape)?;
    }
    Ok(())
}

fn digest_admitted_plan(plan: &DecoderPlan) -> Digest {
    // Hard metadata byte/depth/count admission bounds this one canonical
    // serialization allocation without changing established digest bytes.
    plan.digest()
}

fn copy_region_string(value: &str, field: &str) -> Result<String, String> {
    let mut copy = String::new();
    copy.try_reserve_exact(value.len())
        .map_err(|_| format!("{field} allocation failed"))?;
    copy.push_str(value);
    Ok(copy)
}

fn copy_region_shape(value: &[u64], field: &str) -> Result<Vec<u64>, String> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(value.len())
        .map_err(|_| format!("{field} allocation failed"))?;
    copy.extend_from_slice(value);
    Ok(copy)
}

fn copy_region_digest(value: &Digest, field: &str) -> Result<Digest, String> {
    value
        .try_clone()
        .map_err(|_| format!("{field} allocation failed"))
}

struct FallibleJsonBuffer {
    bytes: Vec<u8>,
}

impl std::io::Write for FallibleJsonBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes
            .try_reserve_exact(bytes.len())
            .map_err(|_| std::io::Error::other("JSON number allocation failed"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn copy_json_number(value: &serde_json::Number, field: &str) -> Result<String, String> {
    let mut output = FallibleJsonBuffer { bytes: Vec::new() };
    serde_json::to_writer(&mut output, value).map_err(|_| format!("{field} allocation failed"))?;
    String::from_utf8(output.bytes).map_err(|_| format!("{field} is not UTF-8"))
}

fn graph(plan: &DecoderPlan, mode: DecoderMode) -> &DecoderGraph {
    match mode {
        DecoderMode::Prefill => &plan.prefill,
        DecoderMode::Decode => &plan.decode,
    }
}

fn operation_source<'a>(
    graph: &'a DecoderGraph,
    input_id: &str,
    operation_id: &str,
    kind: &str,
) -> Result<&'a ModelOperation, String> {
    graph
        .operations
        .iter()
        .find(|candidate| candidate.id == input_id)
        .ok_or_else(|| format!("{kind} operation {operation_id} has missing source {input_id}"))
}

fn exact_attributes<'a>(
    operation: &'a ModelOperation,
    keys: &[&str],
) -> Result<&'a Map<String, Value>, String> {
    let attributes = operation
        .attributes
        .as_object()
        .ok_or_else(|| format!("operation {} attributes are not an object", operation.id))?;
    if attributes.len() != keys.len() || keys.iter().any(|key| !attributes.contains_key(*key)) {
        return Err(format!(
            "operation {} does not have exact semantic attributes",
            operation.id
        ));
    }
    Ok(attributes)
}

fn exact_string<'a>(
    attributes: &'a Map<String, Value>,
    key: &str,
    expected: &str,
    operation: &ModelOperation,
) -> Result<&'a str, String> {
    let actual = attributes.get(key).and_then(Value::as_str);
    if actual != Some(expected) {
        return Err(format!("operation {} has unsupported {key}", operation.id));
    }
    Ok(actual.expect("exact string was checked"))
}

fn nonempty_string_attribute<'a>(
    attributes: &'a Map<String, Value>,
    key: &str,
    operation: &ModelOperation,
) -> Result<&'a str, String> {
    attributes
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("operation {} has invalid {key}", operation.id))
}

fn u64_attribute(
    attributes: &Map<String, Value>,
    key: &str,
    operation: &ModelOperation,
) -> Result<u64, String> {
    attributes
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("operation {} has invalid {key}", operation.id))
}

fn require_attribute_input(
    attributes: &Map<String, Value>,
    key: &str,
    input_id: &str,
    operation: &ModelOperation,
) -> Result<(), String> {
    if attributes.get(key).and_then(Value::as_str) != Some(input_id) {
        return Err(format!(
            "operation {} {key} differs from input provenance",
            operation.id
        ));
    }
    Ok(())
}

fn maximum_position(graph: &DecoderGraph, layer: Option<u64>) -> Result<u32, String> {
    let mut capacities = graph
        .state_outputs
        .iter()
        .filter(|state| {
            state.layer == layer && matches!(state.kind, StateKind::Key | StateKind::Value)
        })
        .map(|state| state.maximum_sequence);
    let capacity = capacities
        .next()
        .ok_or("RoPE layer has no bounded persistent KV state")?;
    if capacity == 0 || capacities.any(|candidate| candidate != capacity) {
        return Err("RoPE layer has inconsistent persistent KV capacities".into());
    }
    u32::try_from(capacity).map_err(|_| "RoPE maximum position exceeds u32".into())
}

fn exact_state<'a>(
    states: &'a [StateTensor],
    id: &str,
    layer: u64,
    kind: StateKind,
    operation_id: &str,
) -> Result<&'a StateTensor, String> {
    let mut matches = states
        .iter()
        .filter(|state| state.id == id && state.layer == Some(layer) && state.kind == kind);
    let state = matches.next().ok_or_else(|| {
        format!("KV cache append {operation_id} has no exact state descriptor for {id}")
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "KV cache append {operation_id} has duplicate state descriptors for {id}"
        ));
    }
    Ok(state)
}

fn exact_state_slot<'a>(
    states: &'a [StateTensor],
    layer: u64,
    kind: StateKind,
    operation_id: &str,
) -> Result<&'a StateTensor, String> {
    let mut matches = states
        .iter()
        .filter(|state| state.layer == Some(layer) && state.kind == kind);
    let state = matches.next().ok_or_else(|| {
        format!("KV cache append {operation_id} has no matching decode state slot")
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "KV cache append {operation_id} has duplicate decode state slots"
        ));
    }
    Ok(state)
}

fn validate_state_descriptor(
    state: &StateTensor,
    shape: &[u64],
    capacity: u64,
    operation: &ModelOperation,
) -> Result<(), String> {
    if state.shape != shape || state.maximum_sequence != capacity {
        return Err(format!(
            "KV cache append {} differs from exact state descriptor",
            operation.id
        ));
    }
    Ok(())
}

fn shape4_usize(shape: &[u64], kind: &str) -> Result<[usize; 4], String> {
    let [batch, heads, sequence, feature] = shape else {
        return Err(format!("{kind} shape must have rank four"));
    };
    let batch = usize::try_from(*batch).map_err(|_| format!("{kind} exceeds usize"))?;
    let heads = usize::try_from(*heads).map_err(|_| format!("{kind} exceeds usize"))?;
    let sequence = usize::try_from(*sequence).map_err(|_| format!("{kind} exceeds usize"))?;
    let feature = usize::try_from(*feature).map_err(|_| format!("{kind} exceeds usize"))?;
    if [batch, heads, sequence, feature].contains(&0) {
        return Err(format!("{kind} dimensions must be positive"));
    }
    shape_elements([batch, heads, sequence, feature], kind)?;
    Ok([batch, heads, sequence, feature])
}

fn shape_elements(shape: [usize; 4], kind: &str) -> Result<usize, String> {
    shape
        .into_iter()
        .try_fold(1_usize, |elements, dimension| {
            elements.checked_mul(dimension)
        })
        .ok_or_else(|| format!("{kind} element count overflows usize"))
}

fn cache_required_bytes(shape: &[u64]) -> Result<usize, String> {
    let shape = shape4_usize(shape, "persistent KV state")?;
    let storage = shape_elements(shape, "persistent KV state")?
        .checked_mul(std::mem::size_of::<i16>())
        .ok_or("KV cache byte count overflows usize")?;
    let lengths = shape[0]
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or("KV cache valid-length byte count overflows usize")?;
    storage
        .checked_add(lengths)
        .ok_or_else(|| "KV cache byte count overflows usize".into())
}

fn view_required_bytes(shape: &[u64]) -> Result<usize, String> {
    let shape = shape4_usize(shape, "KV cache view")?;
    let values = shape_elements(shape, "KV cache view")?
        .checked_mul(std::mem::size_of::<i16>())
        .ok_or("KV cache view byte count overflows usize")?;
    let lengths = shape[0]
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or("KV cache view valid-length byte count overflows usize")?;
    let batch_vectors = shape[0]
        .checked_mul(std::mem::size_of::<Vec<Vec<i16>>>())
        .ok_or("KV cache view batch-vector byte count overflows usize")?;
    let prefix_vectors = shape[0]
        .checked_mul(shape[1])
        .and_then(|count| count.checked_mul(std::mem::size_of::<Vec<i16>>()))
        .ok_or("KV cache view prefix-vector byte count overflows usize")?;
    values
        .checked_add(lengths)
        .and_then(|bytes| bytes.checked_add(batch_vectors))
        .and_then(|bytes| bytes.checked_add(prefix_vectors))
        .ok_or_else(|| "KV cache view byte count overflows usize".into())
}

fn validate_sequential_absolute_positions(
    positions: &[u32],
    shape: [usize; 4],
    maximum_position: u32,
) -> Result<(), String> {
    let [batch, _, sequence, _] = shape;
    let expected_length = batch
        .checked_mul(sequence)
        .ok_or("RoPE positions length overflows usize")?;
    if positions.len() != expected_length {
        return Err(format!(
            "RoPE requires {expected_length} positions, received {}",
            positions.len()
        ));
    }
    for batch_index in 0..batch {
        let row = &positions[batch_index * sequence..(batch_index + 1) * sequence];
        let base = row[0];
        for (sequence_index, &actual) in row.iter().enumerate() {
            let offset = u32::try_from(sequence_index)
                .map_err(|_| "RoPE sequential position offset exceeds u32")?;
            let expected = base
                .checked_add(offset)
                .ok_or("RoPE sequential absolute positions overflow u32")?;
            if actual != expected {
                return Err(format!(
                    "RoPE batch {batch_index} position {sequence_index} must be {expected}, received {actual}"
                ));
            }
            if actual >= maximum_position {
                return Err(format!(
                    "RoPE position {actual} at batch {batch_index}, sequence {sequence_index} is not below {maximum_position}"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{copy_bounded_cache_view, BoundedKvCacheViewQ10};

    #[test]
    fn plaintext_view_zeroization_clears_values_and_sensitive_lengths() {
        let mut view = BoundedKvCacheViewQ10 {
            prefixes: vec![vec![vec![1, -2, 3].into()]],
            valid_lengths: vec![3].into(),
            maximum_sequence: 4,
            head_dim: 1,
        };
        view.zeroize_contents();
        assert!(view.prefixes.is_empty());
        assert!(view.valid_lengths.is_empty());
        assert_eq!(view.maximum_sequence, 0);
        assert_eq!(view.head_dim, 0);
    }

    #[test]
    fn partial_view_copy_error_drops_zeroizing_prefixes() {
        let cache = pllm_core::BoundedKvCacheQ10::initialize(
            1,
            2,
            1,
            2,
            &[1, 2, 3, 4],
            1,
            &[0],
            &[true],
            &[1],
        )
        .unwrap();

        let result = copy_bounded_cache_view(&cache, 1, 2, 1, 2, |copied| {
            if copied == 1 {
                Err("injected view copy failure".into())
            } else {
                Ok(())
            }
        });
        let error = match result {
            Ok(_) => panic!("fault injection unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error, "injected view copy failure");
    }
}
