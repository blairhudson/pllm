use pllm_models::{
    DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator, StateKind, StateTensor,
};
use pllm_types::{canonical_digest, pipeline_digest_bytes, Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

pub const DECODER_RUNTIME_SCHEDULE_SCHEMA_VERSION: &str = "pllm.decoder_runtime_schedule.v2";
const DECODER_RUNTIME_SCHEDULE_DIGEST_DOMAIN: &str = "pllm.decoder_runtime_schedule.v2";
const MAX_WINDOW_STATE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_WINDOW_VIEW_ELEMENTS: u64 = 1 << 24;
const MAX_CONVOLUTION_CHANNELS: u64 = 8192;
const MAX_CONVOLUTION_KERNEL: u64 = 16;
const MAX_CONVOLUTION_TOKENS: u64 = 256;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderRuntimeExecutor {
    ClientLocal,
    ClientLinear,
    RemoteStage,
    VerifiedRemoteStage,
    RemoteGatedBlock,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRuntimeOutput {
    pub operation_id: String,
    pub output_shape: Vec<u64>,
    pub stage_offset: u64,
    pub stage_width: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRuntimeStep {
    pub order: u64,
    pub operation_ids: Vec<String>,
    pub operators: Vec<ModelOperator>,
    pub layer: Option<u64>,
    pub input_ids: Vec<String>,
    pub executor: DecoderRuntimeExecutor,
    pub weight_ids: Vec<String>,
    pub outputs: Vec<DecoderRuntimeOutput>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub terminal_row_only: bool,
}

fn is_false(value: &bool) -> bool {
    !value
}

/// Public graph liveness only. Persistent outputs always demand every row;
/// unknown/layout/state/attention operations stop last-row propagation.
fn terminal_demand(graph: &DecoderGraph) -> BTreeMap<&str, u8> {
    let operations: BTreeMap<_, _> = graph
        .operations
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    let mut demand = BTreeMap::new();
    let mut queue = VecDeque::from([(graph.output.as_str(), 2)]);
    queue.extend(
        graph
            .state_outputs
            .iter()
            .map(|state| (state.id.as_str(), 2)),
    );
    while let Some((id, wanted)) = queue.pop_front() {
        let Some(operation) = operations.get(id) else {
            continue;
        };
        if demand.get(id).copied().unwrap_or(0) >= wanted {
            continue;
        }
        demand.insert(id, wanted);
        let upstream = match operation.operator {
            ModelOperator::LastToken => 1,
            ModelOperator::Linear
            | ModelOperator::OutputHead
            | ModelOperator::RmsNorm
            | ModelOperator::ResidualAdd
            | ModelOperator::Silu
            | ModelOperator::Multiply
            | ModelOperator::GeluTanh
            | ModelOperator::Softcap => wanted,
            _ => 2,
        };
        queue.extend(
            operation
                .inputs
                .iter()
                .map(|input| (input.as_str(), upstream)),
        );
    }
    demand
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRuntimePhaseSchedule {
    pub mode: DecoderMode,
    pub batch: u64,
    pub query_sequence: u64,
    pub maximum_key_sequence: u64,
    pub state_inputs: Vec<StateTensor>,
    pub state_outputs: Vec<StateTensor>,
    pub steps: Vec<DecoderRuntimeStep>,
    pub output: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderRuntimeSchedule {
    pub schema_version: String,
    pub composition_digest: Digest,
    pub model_plan_digest: Digest,
    pub model_config_digest: Digest,
    pub prefill: DecoderRuntimePhaseSchedule,
    pub decode: DecoderRuntimePhaseSchedule,
    pub protected_execution: bool,
    pub complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub private_boundary: Option<crate::decoder_boundary::PrivateBoundaryContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonlinear: Option<crate::masked_wire_schedule::MaskedWireNumericContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_state: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlp_tiling: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_epochs: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inflight_prefix: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_reuse: Option<serde_json::Value>,
}

impl DecoderRuntimeSchedule {
    pub fn digest(&self) -> Digest {
        canonical_digest(DECODER_RUNTIME_SCHEDULE_DIGEST_DOMAIN, self)
    }
}

#[derive(Clone, Copy, Debug)]
struct RemoteMember<'a> {
    operation: &'a ModelOperation,
    index: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct GroupRow {
    last_id_position: usize,
    remote_id_position: Option<usize>,
    start: usize,
    len: usize,
    leader: bool,
    repeated_remote_id: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RemoteGroup<'g, 'a> {
    pub(crate) operations: RemoteMembers<'g, 'a>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RemoteMembers<'g, 'a> {
    members: &'g [RemoteMember<'a>],
}

impl<'g, 'a> RemoteMembers<'g, 'a> {
    pub(crate) fn iter(&self) -> RemoteMemberIter<'g, 'a> {
        RemoteMemberIter {
            remaining: self.members.iter(),
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.members.len()
    }
    pub(crate) fn first(&self) -> &'a ModelOperation {
        self.members[0].operation
    }
}

#[derive(Debug)]
pub(crate) struct RemoteMemberIter<'g, 'a> {
    remaining: std::slice::Iter<'g, RemoteMember<'a>>,
}

impl<'a> Iterator for RemoteMemberIter<'_, 'a> {
    type Item = &'a ModelOperation;
    fn next(&mut self) -> Option<Self::Item> {
        self.remaining.next().map(|member| {
            #[cfg(test)]
            GROUPING_ACCESS.with(|counts| {
                let mut c = counts.get();
                c.member_visits += 1;
                counts.set(c);
            });
            member.operation
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.remaining.size_hint()
    }
}
impl ExactSizeIterator for RemoteMemberIter<'_, '_> {}

/// One indexed owner, shared by ordinary runtime and inventory. No owned keys,
/// String clones, per-group allocation, tree nodes or growing buffers.
#[derive(Debug)]
pub(crate) struct RemoteGroups<'a> {
    graph: &'a DecoderGraph,
    by_id: Box<[usize]>,
    rows: Box<[GroupRow]>,
    members: Box<[RemoteMember<'a>]>,
    groups: usize,
    #[cfg(test)]
    usage: GroupingUsage,
}

impl<'a> RemoteGroups<'a> {
    /// Original ID-based leader lookup at a graph position, including local
    /// operations shadowing remote IDs. The ID-run prepass cached this alias;
    /// neither membership nor dispatch needs another search or allocation.
    pub(crate) fn at(&self, index: usize) -> Option<RemoteGroup<'_, 'a>> {
        #[cfg(test)]
        GROUPING_ACCESS.with(|counts| {
            let mut c = counts.get();
            c.positional_probes += 1;
            counts.set(c);
        });
        let row = &self.rows[self.rows.get(index)?.remote_id_position?];
        row.leader.then(|| RemoteGroup {
            operations: RemoteMembers {
                members: &self.members[row.start..row.start + row.len],
            },
        })
    }
    pub(crate) fn is_remote(&self, index: usize) -> bool {
        self.rows
            .get(index)
            .is_some_and(|row| row.remote_id_position.is_some())
    }
    pub(crate) fn get(&self, id: &str) -> Option<RemoteGroup<'_, 'a>> {
        // Lower bound also preserves legacy behavior for a nonremote ID that
        // shadows a remote leader. Duplicate remote IDs never reach this owner.
        let (mut lo, mut hi) = (0, self.by_id.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            #[cfg(test)]
            GROUPING_ACCESS.with(|counts| {
                let mut c = counts.get();
                c.id_probes += 1;
                counts.set(c);
            });
            if self.graph.operations[self.by_id[mid]].id.as_str() < id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let &index = self.by_id.get(lo)?;
        if self.graph.operations[index].id != id {
            return None;
        }
        self.at(index)
    }
    pub(crate) fn len(&self) -> usize {
        self.groups
    }
    #[cfg(test)]
    pub(crate) fn usage(&self) -> GroupingUsage {
        self.usage
    }
    #[cfg(test)]
    pub(crate) fn retained_index_bytes(&self) -> usize {
        std::mem::size_of_val(self.by_id.as_ref())
            + std::mem::size_of_val(self.rows.as_ref())
            + std::mem::size_of_val(self.members.as_ref())
    }
}

/// Allocation-free diagnostics. Legacy wrappers alone allocate error Strings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RemoteGroupingError<'a> {
    Layers,
    Weight(&'a str),
    LookupHead,
    Duplicate(&'a str),
    Overflow,
    Capacity,
    WorkLimit,
}

/// Logical grouping workspace only. Inputs, callers and downstream lowering are
/// not included. All keys and members are aliases into the immutable graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RemoteGroupingResources {
    pub(crate) workspace_bytes: usize,
    pub(crate) heap_bytes: usize,
    pub(crate) work_bound: u128,
    pub(crate) comparison_bound: u128,
}

/// Construction work units: explicit loop-item visits, scalar key comparisons,
/// conservatively charged string spans (min byte length + termination), heap
/// comparator calls/swaps and weight accesses. Not dependency/allocator CPU or
/// native instructions. String ordering itself reuses optimized standard Ord.
/// `comparisons` counts heap comparator calls only, not charged key byte spans.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GroupingUsage {
    pub(crate) work: u128,
    pub(crate) comparisons: u128,
}
impl GroupingUsage {
    fn step(&mut self) {
        self.work += 1;
    }
}

#[derive(Default)]
struct GroupingSortState {
    root: usize,
    child: usize,
    end: usize,
}

#[derive(Default)]
struct GroupingBuildState {
    usage: GroupingUsage,
    start: usize,
    end: usize,
    first_remote: Option<usize>,
    groups: usize,
    token_lookups: usize,
    output_heads: usize,
    cursor: usize,
    previous_layer: Option<u64>,
    layers: u64,
}

#[derive(Debug)]
pub(crate) struct RemoteGroupingQuote<'a> {
    graph: &'a DecoderGraph,
    remote_count: usize,
    resources: RemoteGroupingResources,
}

#[cfg(test)]
#[derive(Clone, Copy, Default)]
pub(crate) struct GroupingAccess {
    pub(crate) positional_probes: usize,
    pub(crate) id_probes: usize,
    pub(crate) member_visits: usize,
}
#[cfg(test)]
thread_local! { pub(crate) static GROUPING_ACCESS: std::cell::Cell<GroupingAccess> = const { std::cell::Cell::new(GroupingAccess { positional_probes: 0, id_probes: 0, member_visits: 0 }) }; }
#[cfg(test)]
thread_local! { pub(crate) static GROUPING_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

impl fmt::Display for RemoteGroupingError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layers => f.write_str("decoder runtime layers must be contiguous from zero"),
            Self::Weight(id) => write!(
                f,
                "remote operation {id} requires one declared weight artifact"
            ),
            Self::LookupHead => {
                f.write_str("decoder runtime requires token lookup and one output head")
            }
            Self::Duplicate(id) => write!(f, "operation {id} belongs to multiple runtime stages"),
            Self::Overflow => f.write_str("decoder grouping resource arithmetic overflow"),
            Self::Capacity => f.write_str("decoder grouping workspace capacity limit exceeded"),
            Self::WorkLimit => f.write_str("decoder grouping work limit exceeded"),
        }
    }
}

fn remote_operator(operator: ModelOperator) -> bool {
    matches!(
        operator,
        ModelOperator::TokenLookup | ModelOperator::Linear | ModelOperator::OutputHead
    )
}

fn bounded_convolution_operator(operation: &ModelOperation) -> bool {
    let attrs = &operation.attributes;
    let Some(fields) = attrs.as_object() else {
        return false;
    };
    let channels = attrs.get("groups").and_then(serde_json::Value::as_u64);
    let kernel = attrs.get("kernel_size").and_then(serde_json::Value::as_u64);
    let mode = attrs.get("mode").and_then(serde_json::Value::as_str);
    if fields.len() != 6
        || !channels.is_some_and(|width| (1..=MAX_CONVOLUTION_CHANNELS).contains(&width))
        || !kernel.is_some_and(|width| (1..=MAX_CONVOLUTION_KERNEL).contains(&width))
        || attrs
            .get("weight")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        || attrs.get("bias") != Some(&serde_json::Value::Null)
        || attrs.get("activation").and_then(serde_json::Value::as_str) != Some("silu")
        || !matches!(mode, Some("chunk" | "recurrent"))
        || operation.inputs.len() != 2
        || operation.layer.is_none()
    {
        return false;
    }
    let channels = channels.expect("bounded channel count");
    let kernel = kernel.expect("bounded kernel size");
    match operation.operator {
        ModelOperator::CausalConvolution => {
            operation.state_kind.is_none()
                && operation.output_shape.len() == 3
                && operation.output_shape[0] == 1
                && operation.output_shape[2] == channels
                && (1..=MAX_CONVOLUTION_TOKENS).contains(&operation.output_shape[1])
                && operation.output_shape[1]
                    .checked_mul(channels)
                    .is_some_and(|elements| elements <= 1 << 22)
        }
        ModelOperator::ConvolutionStateUpdate => {
            operation.state_kind == Some(StateKind::Convolution)
                && operation.output_shape == [1, channels, kernel]
        }
        _ => false,
    }
}

fn bounded_delta_decay(operation: &ModelOperation) -> bool {
    let attrs = &operation.attributes;
    let Some(fields) = attrs.as_object() else {
        return false;
    };
    fields.len() == 4
        && attrs
            .get("a_log")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| !v.is_empty())
        && attrs
            .get("dt_bias")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| !v.is_empty())
        && attrs["formula"] == "-exp(A_log)*softplus(a+dt_bias)"
        && attrs["compute_dtype"] == "float32"
        && operation.inputs.len() == 1
        && operation.layer.is_some()
        && operation.state_kind.is_none()
        && matches!(operation.output_shape.as_slice(), [1, tokens, heads]
            if (1..=MAX_CONVOLUTION_TOKENS).contains(tokens) && (1..=64).contains(heads))
}

fn bounded_delta_operator(operation: &ModelOperation) -> bool {
    let attrs = &operation.attributes;
    let Some(fields) = attrs.as_object() else {
        return false;
    };
    let Some(scale) = attrs
        .get("query_scale")
        .and_then(serde_json::Value::as_object)
    else {
        return false;
    };
    let denominator = scale
        .get("sqrt_denominator")
        .and_then(serde_json::Value::as_u64);
    let repeats = attrs
        .get("key_head_repeats")
        .and_then(serde_json::Value::as_u64);
    let valid_shape = match operation.operator {
        ModelOperator::GatedDeltaStateUpdate => matches!(operation.output_shape.as_slice(),
            [1, heads, key, value] if (1..=64).contains(heads)
                && (1..=256).contains(key) && (1..=256).contains(value)
                && heads.checked_mul(*key).and_then(|v| v.checked_mul(*value))
                    .is_some_and(|count| count <= 2 * 1024 * 1024)),
        ModelOperator::GatedDeltaRule => matches!(operation.output_shape.as_slice(),
            [1, tokens, heads, value] if (1..=MAX_CONVOLUTION_TOKENS).contains(tokens)
                && (1..=64).contains(heads) && (1..=256).contains(value)),
        _ => false,
    };
    fields.len() == 7
        && scale.len() == 2
        && scale.get("numerator").and_then(serde_json::Value::as_u64) == Some(1)
        && denominator.is_some_and(|width| (1..=256).contains(&width))
        && repeats.is_some_and(|width| (1..=16).contains(&width))
        && attrs["qk_l2_normalize"] == true
        && attrs["normalization_epsilon"] == "0.000001"
        && attrs["chunk_size"] == 64
        && matches!(
            attrs.get("mode").and_then(serde_json::Value::as_str),
            Some("chunked" | "recurrent")
        )
        && attrs["state_update"] == "S'=exp(g)*S+k*(v-S^T*k)*beta"
        && operation.inputs.len() == 6
        && operation.layer.is_some()
        && valid_shape
        && match operation.operator {
            ModelOperator::GatedDeltaStateUpdate => {
                operation.state_kind == Some(StateKind::Recurrent)
            }
            ModelOperator::GatedDeltaRule => operation.state_kind.is_none(),
            _ => false,
        }
}

fn bounded_state_initializer(operation: &ModelOperation) -> bool {
    let state_kind = operation
        .attributes
        .get("state_kind")
        .and_then(serde_json::Value::as_str);
    if operation.operator != ModelOperator::StateInitialize
        || !operation.inputs.is_empty()
        || operation.layer.is_none()
        || operation.state_kind.is_some()
        || !matches!(state_kind, Some("convolution" | "recurrent"))
        || operation.attributes
            != serde_json::json!({
                "initial_value": 0,
                "dtype": "float32",
                "state_kind": state_kind
            })
    {
        return false;
    }
    match (state_kind, operation.output_shape.as_slice()) {
        (Some("convolution"), [batch, channels, kernel]) => {
            *batch == 1
                && (1..=MAX_CONVOLUTION_CHANNELS).contains(channels)
                && (1..=MAX_CONVOLUTION_KERNEL).contains(kernel)
        }
        (Some("recurrent"), [batch, heads, key, value]) => {
            *batch == 1
                && (1..=64).contains(heads)
                && (1..=256).contains(key)
                && (1..=256).contains(value)
                && heads
                    .checked_mul(*key)
                    .and_then(|count| count.checked_mul(*value))
                    .is_some_and(|count| count <= 2 * 1024 * 1024)
        }
        _ => false,
    }
}

fn validate_causal_convolution_contract(graph: &DecoderGraph) -> Result<(), String> {
    let convolutions: Vec<_> = graph
        .operations
        .iter()
        .filter(|row| row.operator == ModelOperator::CausalConvolution)
        .collect();
    let updates: Vec<_> = graph
        .operations
        .iter()
        .filter(|row| row.operator == ModelOperator::ConvolutionStateUpdate)
        .collect();
    if convolutions.len() != updates.len() {
        return Err("causal convolution lacks an exact state-update partner".into());
    }
    let expected_mode = match graph.mode {
        DecoderMode::Prefill => "chunk",
        DecoderMode::Decode => "recurrent",
    };
    for convolution in convolutions {
        if !bounded_convolution_operator(convolution) {
            return Err("causal convolution numeric contract is unsupported".into());
        }
        let state = graph
            .state_inputs
            .iter()
            .find(|state| state.id == convolution.inputs[1]);
        let initialized = graph
            .operations
            .iter()
            .find(|row| row.id == convolution.inputs[1]);
        let paired: Vec<_> = updates
            .iter()
            .filter(|row| row.inputs == convolution.inputs && row.layer == convolution.layer)
            .collect();
        if paired.len() != 1 {
            return Err("causal convolution has no unique state update".into());
        }
        let update = paired[0];
        let result = graph
            .state_outputs
            .iter()
            .find(|row| row.id == update.id)
            .ok_or("causal convolution state has no persistent output")?;
        let channels = convolution.attributes["groups"]
            .as_u64()
            .ok_or("causal convolution channel count is invalid")?;
        let kernel = convolution.attributes["kernel_size"]
            .as_u64()
            .ok_or("causal convolution kernel size is invalid")?;
        let projection = graph
            .operations
            .iter()
            .find(|row| row.id == convolution.inputs[0])
            .ok_or("causal convolution has no declared projection")?;
        if !bounded_convolution_operator(convolution)
            || !bounded_convolution_operator(update)
            || convolution.attributes != update.attributes
            || convolution.attributes["mode"] != expected_mode
            || convolution.output_shape != [1, graph.query_sequence, channels]
            || projection.output_shape != convolution.output_shape
            || result.kind != StateKind::Convolution
            || result.layer != convolution.layer
            || result.maximum_sequence != kernel
            || update.output_shape != result.shape
            || result.shape != [1, channels, kernel]
        {
            return Err("causal convolution and state shape or numeric contract differ".into());
        }
        match graph.mode {
            DecoderMode::Prefill => {
                let initializer =
                    initialized.ok_or("causal convolution prefill omits zero state")?;
                if state.is_some()
                    || !bounded_state_initializer(initializer)
                    || initializer.attributes["state_kind"] != "convolution"
                    || initializer.layer != convolution.layer
                    || initializer.output_shape != result.shape
                {
                    return Err(
                        "causal convolution prefill state is not explicitly initialized".into(),
                    );
                }
            }
            DecoderMode::Decode => {
                let prior = state.ok_or("causal convolution decode lacks retained state")?;
                if initialized.is_some()
                    || prior.kind != StateKind::Convolution
                    || prior.layer != convolution.layer
                    || prior.shape != result.shape
                    || prior.maximum_sequence != kernel
                {
                    return Err(
                        "causal convolution decode state does not match its producer".into(),
                    );
                }
            }
        }
    }
    let bytes = graph
        .state_outputs
        .iter()
        .filter(|state| matches!(state.kind, StateKind::Convolution | StateKind::Recurrent))
        .try_fold(0_u64, |total, state| {
            let size = state
                .shape
                .iter()
                .try_fold(4_u64, |product, width| product.checked_mul(*width))?;
            total.checked_add(size)
        })
        .ok_or("hybrid state capacity overflowed")?;
    if bytes > MAX_WINDOW_STATE_BYTES {
        return Err("hybrid decoder state exceeds the client memory ceiling".into());
    }
    Ok(())
}

fn validate_gated_delta_contract(graph: &DecoderGraph) -> Result<(), String> {
    let rules: Vec<_> = graph
        .operations
        .iter()
        .filter(|row| row.operator == ModelOperator::GatedDeltaRule)
        .collect();
    let updates: Vec<_> = graph
        .operations
        .iter()
        .filter(|row| row.operator == ModelOperator::GatedDeltaStateUpdate)
        .collect();
    if rules.len() != updates.len() {
        return Err("gated-delta rule lacks an exact state-update partner".into());
    }
    let expected_mode = match graph.mode {
        DecoderMode::Prefill => "chunked",
        DecoderMode::Decode => "recurrent",
    };
    for rule in rules {
        if !bounded_delta_operator(rule) {
            return Err("gated-delta numeric contract is unsupported".into());
        }
        let paired: Vec<_> = updates
            .iter()
            .filter(|row| row.inputs == rule.inputs && row.layer == rule.layer)
            .collect();
        if paired.len() != 1 {
            return Err("gated-delta rule has no unique state update".into());
        }
        let update = paired[0];
        let [_, batch_tokens, value_heads, value_dim] = rule.output_shape.as_slice() else {
            return Err("gated-delta output shape is invalid".into());
        };
        let [_, state_heads, key_dim, state_value_dim] = update.output_shape.as_slice() else {
            return Err("gated-delta state shape is invalid".into());
        };
        let repeats = rule.attributes["key_head_repeats"]
            .as_u64()
            .ok_or("gated-delta head ratio is invalid")?;
        let key_heads = value_heads
            .checked_div(repeats)
            .filter(|heads| *heads > 0 && heads * repeats == *value_heads)
            .ok_or("gated-delta head geometry is invalid")?;
        let expected_width = key_heads
            .checked_mul(*key_dim)
            .ok_or("gated-delta key width overflowed")?;
        let expected_value_width = value_heads
            .checked_mul(*value_dim)
            .ok_or("gated-delta value width overflowed")?;
        let producers: Vec<_> = rule.inputs[..5]
            .iter()
            .map(|id| graph.operations.iter().find(|row| &row.id == id))
            .collect();
        if producers.iter().any(Option::is_none) {
            return Err("gated-delta input has no declared producer".into());
        }
        let producer = |index: usize| producers[index].expect("all producers were checked");
        let state = graph
            .state_inputs
            .iter()
            .find(|row| row.id == rule.inputs[5]);
        let initialized = graph.operations.iter().find(|row| row.id == rule.inputs[5]);
        let output = graph
            .state_outputs
            .iter()
            .find(|row| row.id == update.id)
            .ok_or("gated-delta state has no persistent output")?;
        if !bounded_delta_operator(update)
            || update.attributes != rule.attributes
            || rule.attributes["mode"] != expected_mode
            || *batch_tokens != graph.query_sequence
            || *value_heads != *state_heads
            || *value_dim != *state_value_dim
            || rule.attributes["query_scale"]["sqrt_denominator"] != *key_dim
            || producer(0).output_shape != [1, *batch_tokens, expected_width]
            || producer(1).output_shape != producer(0).output_shape
            || producer(2).output_shape != [1, *batch_tokens, expected_value_width]
            || producer(3).operator != ModelOperator::GatedDeltaDecay
            || !bounded_delta_decay(producer(3))
            || producer(4).operator != ModelOperator::Sigmoid
            || producer(4).attributes != serde_json::json!({})
            || producer(3).output_shape != [1, *batch_tokens, *value_heads]
            || producer(4).output_shape != producer(3).output_shape
            || output.kind != StateKind::Recurrent
            || output.layer != rule.layer
            || output.maximum_sequence != 1
            || output.shape != update.output_shape
        {
            return Err(
                "gated-delta operators or retained state violate their declared numeric contract"
                    .into(),
            );
        }
        match graph.mode {
            DecoderMode::Prefill => {
                let initial = initialized.ok_or("gated-delta prefill omits zero state")?;
                if state.is_some()
                    || !bounded_state_initializer(initial)
                    || initial.attributes["state_kind"] != "recurrent"
                    || initial.layer != rule.layer
                    || initial.output_shape != output.shape
                {
                    return Err("gated-delta prefill state is not explicitly initialized".into());
                }
            }
            DecoderMode::Decode => {
                let prior = state.ok_or("gated-delta decode lacks retained state")?;
                if initialized.is_some()
                    || prior.kind != StateKind::Recurrent
                    || prior.layer != rule.layer
                    || prior.shape != output.shape
                    || prior.maximum_sequence != 1
                {
                    return Err("gated-delta decode state does not match its producer".into());
                }
            }
        }
    }
    Ok(())
}

fn validate_causal_convolution_handoff(plan: &DecoderPlan) -> Result<(), String> {
    let slots = |states: &[StateTensor]| -> Result<BTreeMap<u64, (Vec<u64>, u64)>, String> {
        let mut bound = BTreeMap::new();
        for state in states
            .iter()
            .filter(|row| row.kind == StateKind::Convolution)
        {
            let layer = state.layer.ok_or("convolution state lacks a layer")?;
            if bound
                .insert(layer, (state.shape.clone(), state.maximum_sequence))
                .is_some()
            {
                return Err("convolution state has duplicate layer ownership".into());
            }
        }
        Ok(bound)
    };
    if slots(&plan.prefill.state_outputs)? != slots(&plan.decode.state_inputs)? {
        return Err("convolution prefill and decode state contracts differ".into());
    }
    let kernels = |graph: &DecoderGraph| -> Result<BTreeMap<u64, serde_json::Map<String, serde_json::Value>>, String> {
        let mut bound = BTreeMap::new();
        for operation in graph
            .operations
            .iter()
            .filter(|row| row.operator == ModelOperator::CausalConvolution)
        {
            let layer = operation.layer.ok_or("causal convolution lacks a layer")?;
            let mut attrs = operation
                .attributes
                .as_object()
                .ok_or("causal convolution has no numeric descriptor")?
                .clone();
            attrs.remove("mode");
            if bound.insert(layer, attrs).is_some() {
                return Err("causal convolution has duplicate layer ownership".into());
            }
        }
        Ok(bound)
    };
    if kernels(&plan.prefill)? != kernels(&plan.decode)? {
        return Err("convolution prefill and decode kernels differ".into());
    }
    let recurrent_slots =
        |states: &[StateTensor]| -> Result<BTreeMap<u64, (Vec<u64>, u64)>, String> {
            let mut bound = BTreeMap::new();
            for state in states.iter().filter(|row| row.kind == StateKind::Recurrent) {
                let layer = state.layer.ok_or("gated-delta state lacks a layer")?;
                if bound
                    .insert(layer, (state.shape.clone(), state.maximum_sequence))
                    .is_some()
                {
                    return Err("gated-delta state has duplicate layer ownership".into());
                }
            }
            Ok(bound)
        };
    if recurrent_slots(&plan.prefill.state_outputs)? != recurrent_slots(&plan.decode.state_inputs)?
    {
        return Err("gated-delta prefill and decode state contracts differ".into());
    }
    for operator in [
        ModelOperator::GatedDeltaRule,
        ModelOperator::GatedDeltaDecay,
    ] {
        let attrs_by_layer = |graph: &DecoderGraph| -> Result<
            BTreeMap<u64, serde_json::Map<String, serde_json::Value>>,
            String,
        > {
            let mut bound = BTreeMap::new();
            for row in graph
                .operations
                .iter()
                .filter(|row| row.operator == operator)
            {
                let layer = row.layer.ok_or("gated-delta operator lacks a layer")?;
                let mut attrs = row
                    .attributes
                    .as_object()
                    .ok_or("gated-delta attrs are invalid")?
                    .clone();
                attrs.remove("mode");
                if bound.insert(layer, attrs).is_some() {
                    return Err("gated-delta operator repeats its layer".into());
                }
            }
            Ok(bound)
        };
        if attrs_by_layer(&plan.prefill)? != attrs_by_layer(&plan.decode)? {
            return Err("gated-delta coefficients or rule change across decoder phases".into());
        }
    }
    Ok(())
}

fn local_operator(operation: &ModelOperation) -> bool {
    let bfloat16 = |value: &serde_json::Value| {
        value
            .get("compute_dtype")
            .and_then(serde_json::Value::as_str)
            == Some("bfloat16")
            && value
                .get("output_dtype")
                .and_then(serde_json::Value::as_str)
                == Some("bfloat16")
    };
    match operation.operator {
        ModelOperator::StateInitialize => bounded_state_initializer(operation),
        ModelOperator::Sigmoid => {
            operation.attributes == serde_json::json!({})
                && operation.inputs.len() == 1
                && matches!(operation.output_shape.as_slice(), [1, tokens, width]
                    if (1..=MAX_CONVOLUTION_TOKENS).contains(tokens)
                        && (1..=8192).contains(width))
        }
        ModelOperator::GatedDeltaDecay => bounded_delta_decay(operation),
        ModelOperator::GatedDeltaRule | ModelOperator::GatedDeltaStateUpdate => {
            bounded_delta_operator(operation)
        }
        ModelOperator::RmsNormGated => {
            let attrs = &operation.attributes;
            attrs.as_object().is_some_and(|fields| fields.len() == 5)
                && attrs
                    .get("epsilon")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|value| value.parse::<f64>().ok())
                    .is_some_and(|epsilon| epsilon.is_finite() && epsilon > 0.0 && epsilon <= 0.01)
                && attrs
                    .get("weight")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|weight| !weight.is_empty())
                && attrs["activation"] == "silu"
                && attrs["weight_offset"] == 0
                && attrs["norm_before_gate"] == true
                && operation.inputs.len() == 2
                && operation.layer.is_some()
                && operation.state_kind.is_none()
                && matches!(operation.output_shape.as_slice(), [1, tokens, heads, value]
                    if (1..=MAX_CONVOLUTION_TOKENS).contains(tokens)
                        && (1..=64).contains(heads) && (1..=256).contains(value))
        }
        ModelOperator::Reshape
        | ModelOperator::KvCacheAppend
        | ModelOperator::Silu
        | ModelOperator::LastToken
        | ModelOperator::GreedyTokenSelection
        | ModelOperator::TokenFeedback => true,
        ModelOperator::ResidualAdd | ModelOperator::Multiply => {
            let attrs = &operation.attributes;
            attrs.as_object().is_some_and(serde_json::Map::is_empty)
                || attrs
                    .get("output_dtype")
                    .and_then(serde_json::Value::as_str)
                    == Some("bfloat16")
        }
        ModelOperator::Softmax => {
            let attrs = &operation.attributes;
            attrs.as_object().is_some_and(|fields| {
                fields.is_empty()
                    || (fields.len() == 1
                        && attrs.get("axis").and_then(serde_json::Value::as_i64) == Some(-1))
            }) || (attrs.as_object().is_some_and(|fields| fields.len() == 2)
                && attrs.get("axis").and_then(serde_json::Value::as_i64) == Some(-1)
                && attrs
                    .get("compute_dtype")
                    .and_then(serde_json::Value::as_str)
                    == Some("float32"))
                || (attrs
                    .get("output_dtype")
                    .and_then(serde_json::Value::as_str)
                    == Some("bfloat16")
                    && attrs
                        .get("compute_dtype")
                        .and_then(serde_json::Value::as_str)
                        == Some("float32")
                    && attrs.get("axis").and_then(serde_json::Value::as_i64) == Some(-1))
        }
        ModelOperator::RmsNorm => {
            let attrs = &operation.attributes;
            if attrs.get("output_dtype").is_none() && attrs.get("compute_dtype").is_none() {
                return true;
            }
            let weight = attrs.get("weight").and_then(serde_json::Value::as_str);
            let offset = attrs
                .get("weight_offset")
                .and_then(serde_json::Value::as_i64);
            attrs
                .get("output_dtype")
                .and_then(serde_json::Value::as_str)
                == Some("bfloat16")
                && attrs
                    .get("compute_dtype")
                    .and_then(serde_json::Value::as_str)
                    == Some("float32")
                && attrs
                    .get("epsilon")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|value| !value.is_empty())
                && attrs.get("with_scale").and_then(serde_json::Value::as_bool)
                    == Some(weight.is_some())
                && matches!(offset, Some(0 | 1))
                && (weight.is_some() || offset == Some(0))
        }
        ModelOperator::CacheSuffix => {
            let attrs = &operation.attributes;
            if attrs.get("axis").and_then(serde_json::Value::as_u64) == Some(2)
                && attrs.get("semantics").and_then(serde_json::Value::as_str)
                    == Some("visible_valid_prefix")
            {
                return true;
            }
            attrs.get("axis").and_then(serde_json::Value::as_u64) == Some(3)
                && attrs.get("output_axis").and_then(serde_json::Value::as_u64) == Some(2)
                && attrs.get("semantics").and_then(serde_json::Value::as_str)
                    == Some("persist_last_valid_past_tokens")
                && attrs
                    .get("absolute_write_positions_input")
                    .and_then(serde_json::Value::as_str)
                    == Some("input.positions")
                && attrs
                    .get("padding_mask_input")
                    .and_then(serde_json::Value::as_str)
                    == Some("input.attention_mask")
                && attrs
                    .get("valid_lengths_input")
                    .and_then(serde_json::Value::as_str)
                    == Some("input.sequence_lengths")
                && operation.inputs.len() == 4
                && operation.inputs[1..]
                    == [
                        "input.positions",
                        "input.attention_mask",
                        "input.sequence_lengths",
                    ]
                && operation.output_shape.len() == 4
                && attrs
                    .get("maximum_sequence")
                    .and_then(serde_json::Value::as_u64)
                    == Some(operation.output_shape[2])
                && matches!(
                    operation.state_kind.as_ref(),
                    Some(StateKind::Key | StateKind::Value)
                )
        }
        ModelOperator::AttentionScores | ModelOperator::AttentionValues => {
            let key = if operation.operator == ModelOperator::AttentionScores {
                "key_layout"
            } else {
                "value_layout"
            };
            let attrs = &operation.attributes;
            match attrs.get(key).and_then(serde_json::Value::as_str) {
                None => {
                    attrs.get("output_dtype").is_none()
                        && attrs
                            .get("group_size")
                            .and_then(serde_json::Value::as_u64)
                            .is_some_and(|value| value > 0)
                }
                Some("batch_kv_heads_sequence_feature" | "batch_kv_heads_query_window_feature") => {
                    attrs
                        .get("group_size")
                        .and_then(serde_json::Value::as_u64)
                        .is_some_and(|value| value > 0)
                        && attrs
                            .get("output_dtype")
                            .and_then(serde_json::Value::as_str)
                            == Some("bfloat16")
                        && operation.output_shape.len() == 4
                        && (operation.operator != ModelOperator::AttentionScores
                            || (operation.layer.is_some_and(|layer| {
                                attrs
                                    .get("key_value_source_layer")
                                    .and_then(serde_json::Value::as_u64)
                                    .is_some_and(|source| source <= layer)
                            })))
                }
                _ => false,
            }
        }
        ModelOperator::CausalMask => {
            let attrs = &operation.attributes;
            if attrs.as_object().is_some_and(serde_json::Map::is_empty) {
                return true;
            }
            if attrs
                .get("absolute_positions_input")
                .and_then(serde_json::Value::as_str)
                != Some("input.positions")
                || attrs
                    .get("padding_mask_input")
                    .and_then(serde_json::Value::as_str)
                    != Some("input.attention_mask")
                || attrs
                    .get("valid_lengths_input")
                    .and_then(serde_json::Value::as_str)
                    != Some("input.sequence_lengths")
            {
                return false;
            }
            match attrs.get("kind").and_then(serde_json::Value::as_str) {
                Some("full_causal") => {
                    attrs
                        .get("maximum_position_embeddings")
                        .and_then(serde_json::Value::as_u64)
                        .is_some_and(|value| value > 0)
                        && attrs.get("key_domain").and_then(serde_json::Value::as_str)
                            == Some("fixed_capacity")
                        && attrs
                            .get("cache_validity")
                            .and_then(serde_json::Value::as_str)
                            == Some("valid_lengths_fixed_capacity")
                }
                Some("sliding_causal") => {
                    let window = attrs
                        .get("sliding_window")
                        .and_then(serde_json::Value::as_u64);
                    window.is_some_and(|value| {
                        value > 0
                            && attrs
                                .get("left_context")
                                .and_then(serde_json::Value::as_u64)
                                == Some(value - 1)
                    }) && attrs
                        .get("includes_current")
                        .and_then(serde_json::Value::as_bool)
                        == Some(true)
                        && attrs.get("key_domain").and_then(serde_json::Value::as_str)
                            == Some("query_relative_window")
                        && attrs
                            .get("cache_validity")
                            .and_then(serde_json::Value::as_str)
                            == Some("valid_lengths_bounded_suffix")
                }
                _ => false,
            }
        }
        ModelOperator::Scale | ModelOperator::AttentionScale => {
            if operation.operator == ModelOperator::AttentionScale
                && operation.attributes.get("factor").is_none()
            {
                return operation
                    .attributes
                    .get("head_dim")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|value| value > 0);
            }
            let Some(factor) = operation.attributes.get("factor") else {
                return false;
            };
            if !bfloat16(factor)
                || factor
                    .get("factor_rounding_dtype")
                    .and_then(serde_json::Value::as_str)
                    != Some("bfloat16")
            {
                return false;
            }
            match factor.get("kind").and_then(serde_json::Value::as_str) {
                Some("sqrt" | "inverse_sqrt") => {
                    factor
                        .get("radicand")
                        .and_then(serde_json::Value::as_u64)
                        .is_some_and(|value| value > 0)
                        && matches!(
                            factor
                                .get("factor_source_dtype")
                                .and_then(serde_json::Value::as_str),
                            Some("float32_buffer" | "python_float64")
                        )
                }
                Some("rational") => {
                    factor
                        .get("numerator")
                        .and_then(serde_json::Value::as_i64)
                        .is_some()
                        && factor
                            .get("denominator")
                            .and_then(serde_json::Value::as_i64)
                            .is_some_and(|value| value > 0)
                        && factor
                            .get("factor_source_dtype")
                            .and_then(serde_json::Value::as_str)
                            == Some("exact_integer")
                }
                Some("checkpoint_scalar") => {
                    factor
                        .get("weight")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|value| !value.is_empty())
                        && factor.get("weight_shape") == Some(&serde_json::json!([1]))
                        && factor
                            .get("factor_source_dtype")
                            .and_then(serde_json::Value::as_str)
                            == Some("bfloat16")
                }
                _ => false,
            }
        }
        ModelOperator::GeluTanh => {
            bfloat16(&operation.attributes)
                && operation
                    .attributes
                    .get("approximation")
                    .and_then(serde_json::Value::as_str)
                    == Some("tanh")
        }
        ModelOperator::Softcap => {
            bfloat16(&operation.attributes)
                && operation
                    .attributes
                    .get("cap")
                    .and_then(serde_json::Value::as_i64)
                    .is_some_and(|value| value > 0)
                && operation
                    .attributes
                    .get("formula")
                    .and_then(serde_json::Value::as_str)
                    == Some("cap*tanh(input/cap)")
        }
        ModelOperator::Permute => {
            operation.attributes.get("permutation") == Some(&serde_json::json!([0, 2, 1, 3]))
                && operation.output_shape.len() == 4
        }
        ModelOperator::Slice => {
            let start = operation
                .attributes
                .get("start")
                .and_then(serde_json::Value::as_u64);
            let end = operation
                .attributes
                .get("end")
                .and_then(serde_json::Value::as_u64);
            let axis = operation
                .attributes
                .get("axis")
                .and_then(serde_json::Value::as_i64);
            let squeeze = operation
                .attributes
                .get("squeeze")
                .and_then(serde_json::Value::as_bool);
            matches!(operation.output_shape.len(), 3 | 4)
                && match (start, end, axis, squeeze) {
                    (Some(start), Some(end), Some(2), Some(true)) => {
                        operation.output_shape.len() == 3 && start.checked_add(1) == Some(end)
                    }
                    (Some(start), Some(end), Some(-1), Some(false)) => end > start,
                    _ => false,
                }
        }
        ModelOperator::CausalConvolution | ModelOperator::ConvolutionStateUpdate => {
            bounded_convolution_operator(operation)
        }
        ModelOperator::RotaryEmbedding => {
            let attrs = &operation.attributes;
            if ["mrope_interleaved", "mrope_section"]
                .iter()
                .any(|field| attrs.get(*field).is_some())
            {
                let Some(sections) = attrs
                    .get("mrope_section")
                    .and_then(serde_json::Value::as_array)
                else {
                    return false;
                };
                let [1, heads, tokens, width] = operation.output_shape.as_slice() else {
                    return false;
                };
                let Some(rotary) = attrs
                    .get("rotary_dimensions")
                    .and_then(serde_json::Value::as_u64)
                else {
                    return false;
                };
                let Some(theta) = attrs.get("theta").and_then(serde_json::Value::as_f64) else {
                    return false;
                };
                let factor = attrs
                    .get("partial_rotary_factor")
                    .and_then(serde_json::Value::as_str);
                let scaled = match factor {
                    Some("0.25") if width % 4 == 0 => width / 4,
                    Some("1") => *width,
                    _ => return false,
                };
                return attrs.as_object().is_some_and(|fields| fields.len() == 7)
                    && attrs["rope_type"] == "default"
                    && attrs["mrope_interleaved"] == true
                    && attrs["position_policy"] == "text_replicated_axes"
                    && operation.inputs.len() == 2
                    && operation.inputs[1] == "input.positions"
                    && (1..=64).contains(heads)
                    && (1..=MAX_CONVOLUTION_TOKENS).contains(tokens)
                    && (2..=256).contains(width)
                    && scaled == rotary
                    && rotary % 2 == 0
                    && theta.is_finite()
                    && (1.0..=1e12).contains(&theta)
                    && sections.len() == 3
                    && sections.iter().all(|value| value.as_u64().is_some())
                    && sections
                        .iter()
                        .try_fold(0_u64, |sum, item| sum.checked_add(item.as_u64()?))
                        == Some(rotary / 2);
            }
            let scaling = attrs
                .get("frequency_scaling")
                .and_then(|scale| scale.get("kind"))
                .and_then(serde_json::Value::as_str);
            match attrs.get("rope_type").and_then(serde_json::Value::as_str) {
                None => matches!(scaling, None | Some("wavelength_transition")),
                Some("longrope") => scaling == Some("per_frequency_context"),
                Some("default" | "proportional") => scaling.is_none(),
                _ => false,
            }
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum NumericDomain {
    Bfloat16,
    MaskedBfloat16,
    Integer,
}

/// Mixed-precision plans must bind every semantic numeric edge, not merely
/// round individual activations. The client may represent BF16 as float32
/// storage only when every producer and consumer preserves its boundary.
fn bind_bfloat16_graph(graph: &DecoderGraph) -> Result<(), String> {
    let declares_bfloat16 = graph.operations.iter().any(|operation| {
        let attrs = &operation.attributes;
        ["output_dtype", "input_dtype", "compute_dtype"]
            .iter()
            .any(|field| attrs.get(*field).and_then(serde_json::Value::as_str) == Some("bfloat16"))
            || attrs
                .get("factor")
                .and_then(|factor| factor.get("output_dtype"))
                .and_then(serde_json::Value::as_str)
                == Some("bfloat16")
    });
    if !declares_bfloat16 {
        return Ok(());
    }
    let mut domains = BTreeMap::new();
    domains.insert("input.tokens".to_owned(), NumericDomain::Integer);
    domains.insert("input.positions".to_owned(), NumericDomain::Integer);
    domains.insert("input.sequence_lengths".to_owned(), NumericDomain::Integer);
    domains.insert("input.attention_mask".to_owned(), NumericDomain::Integer);
    for state in &graph.state_inputs {
        domains.insert(state.id.clone(), NumericDomain::Bfloat16);
    }
    for operation in &graph.operations {
        let inputs: Vec<_> = operation
            .inputs
            .iter()
            .map(|id| domains.get(id).copied())
            .collect();
        let uses = |indices: &[usize], domain| {
            indices
                .iter()
                .all(|index| inputs.get(*index) == Some(&Some(domain)))
        };
        let attrs = &operation.attributes;
        let output = attrs
            .get("output_dtype")
            .and_then(serde_json::Value::as_str)
            == Some("bfloat16");
        let factor_output = attrs
            .get("factor")
            .and_then(|factor| factor.get("output_dtype"))
            .and_then(serde_json::Value::as_str)
            == Some("bfloat16");
        let result = match operation.operator {
            ModelOperator::TokenLookup if uses(&[0], NumericDomain::Integer) && output => {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::Linear
                if uses(&[0], NumericDomain::Bfloat16)
                    && attrs.get("input_dtype").and_then(serde_json::Value::as_str)
                        == Some("bfloat16")
                    && output =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::OutputHead
            | ModelOperator::RmsNorm
            | ModelOperator::RotaryEmbedding
            | ModelOperator::GeluTanh
            | ModelOperator::Softcap
                if uses(&[0], NumericDomain::Bfloat16) && output =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::Scale | ModelOperator::AttentionScale
                if uses(&[0], NumericDomain::Bfloat16) && factor_output =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::AttentionScores
            | ModelOperator::AttentionValues
            | ModelOperator::ResidualAdd
            | ModelOperator::Multiply
                if uses(&[0, 1], NumericDomain::Bfloat16) && output =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::Reshape
            | ModelOperator::Permute
            | ModelOperator::Slice
            | ModelOperator::LastToken
            | ModelOperator::CacheSuffix
                if uses(&[0], NumericDomain::Bfloat16) =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::KvCacheAppend
                if operation.inputs.iter().enumerate().any(|(index, id)| {
                    id != "input.positions"
                        && id != "input.attention_mask"
                        && id != "input.sequence_lengths"
                        && inputs[index] == Some(NumericDomain::Bfloat16)
                }) && operation.inputs.iter().enumerate().all(|(index, id)| {
                    matches!(
                        id.as_str(),
                        "input.positions" | "input.attention_mask" | "input.sequence_lengths"
                    ) || inputs[index] == Some(NumericDomain::Bfloat16)
                }) =>
            {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::CausalMask if uses(&[0], NumericDomain::Bfloat16) => {
                Some(NumericDomain::MaskedBfloat16)
            }
            ModelOperator::Softmax if uses(&[0], NumericDomain::MaskedBfloat16) && output => {
                Some(NumericDomain::Bfloat16)
            }
            ModelOperator::GreedyTokenSelection if uses(&[0], NumericDomain::Bfloat16) => {
                Some(NumericDomain::Integer)
            }
            ModelOperator::TokenFeedback if uses(&[0], NumericDomain::Integer) => {
                Some(NumericDomain::Integer)
            }
            _ => None,
        };
        let Some(result) = result else {
            return Err(format!(
                "BF16 numeric edge at {} has no complete operator contract",
                operation.id
            ));
        };
        if domains.insert(operation.id.clone(), result).is_some() {
            return Err(format!(
                "BF16 operation {} was declared twice",
                operation.id
            ));
        }
    }
    Ok(())
}

fn weight_id(operation: &ModelOperation) -> Result<String, String> {
    operation
        .attributes
        .get("weight")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "remote operation {} requires one declared weight artifact",
                operation.id
            )
        })
}

fn last_width(operation: &ModelOperation) -> Result<u64, String> {
    operation
        .output_shape
        .last()
        .copied()
        .filter(|width| *width > 0)
        .ok_or_else(|| format!("operation {} has no output width", operation.id))
}

fn grouping_add(a: u128, b: u128) -> Result<u128, RemoteGroupingError<'static>> {
    a.checked_add(b).ok_or(RemoteGroupingError::Overflow)
}
fn grouping_mul(a: u128, b: u128) -> Result<u128, RemoteGroupingError<'static>> {
    a.checked_mul(b).ok_or(RemoteGroupingError::Overflow)
}

fn grouping_layout<T>(count: usize) -> Result<usize, RemoteGroupingError<'static>> {
    std::alloc::Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| RemoteGroupingError::Overflow)
}

struct IndexAllocation<T> {
    slots: Box<[std::mem::MaybeUninit<T>]>,
    initialized: usize,
}
fn grouping_box<T: Copy>(
    count: usize,
    mut make: impl FnMut(usize) -> T,
) -> Result<Box<[T]>, RemoteGroupingError<'static>> {
    grouping_layout::<T>(count)?;
    #[cfg(test)]
    GROUPING_ALLOCATIONS.with(|count| count.set(count.get() + 1));
    let mut allocation = IndexAllocation {
        slots: Box::<[T]>::new_uninit_slice(count),
        initialized: 0,
    };
    while allocation.initialized < count {
        allocation.slots[allocation.initialized].write(make(allocation.initialized));
        allocation.initialized += 1;
    }
    // SAFETY: the entire exact-length slice was initialized. T: Copy has no
    // destructor, so an initializer panic cannot strand an owned element.
    Ok(unsafe { allocation.slots.assume_init() })
}

#[derive(Default)]
struct KeyCompareState {
    input: usize,
}

fn grouping_bytes_cmp(a: &str, b: &str, usage: &mut GroupingUsage) -> std::cmp::Ordering {
    usage.work += a.len().min(b.len()) as u128 + 1;
    a.cmp(b)
}

/// Exactly the original RemoteGroupKey ordering, over original borrowed strings.
fn remote_key_cmp(
    a: &ModelOperation,
    b: &ModelOperation,
    usage: &mut GroupingUsage,
) -> std::cmp::Ordering {
    usage.step();
    let order = a.operator.cmp(&b.operator);
    if !order.is_eq() {
        return order;
    }
    usage.step();
    let order = a.layer.cmp(&b.layer);
    if !order.is_eq() {
        return order;
    }
    let mut cursor = KeyCompareState::default();
    while cursor.input < a.inputs.len().min(b.inputs.len()) {
        let order = grouping_bytes_cmp(&a.inputs[cursor.input], &b.inputs[cursor.input], usage);
        if !order.is_eq() {
            return order;
        }
        cursor.input += 1;
    }
    usage.step();
    let order = a.inputs.len().cmp(&b.inputs.len());
    if !order.is_eq() {
        return order;
    }
    usage.step();
    if a.operator == ModelOperator::OutputHead {
        grouping_bytes_cmp(&a.id, &b.id, usage)
    } else {
        std::cmp::Ordering::Equal
    }
}

/// In-place heapsort: no dependency sort scratch or undocumented comparison
/// constant. At most 2 comparisons per sift level; no recursion or heap growth.
fn grouping_sort<T>(
    items: &mut [T],
    usage: &mut GroupingUsage,
    mut compare: impl FnMut(&T, &T, &mut GroupingUsage) -> std::cmp::Ordering,
) {
    fn sift<T>(
        items: &mut [T],
        state: &mut GroupingSortState,
        usage: &mut GroupingUsage,
        compare: &mut impl FnMut(&T, &T, &mut GroupingUsage) -> std::cmp::Ordering,
    ) {
        while state.root < state.end / 2 {
            state.child = 2 * state.root + 1;
            if state.child + 1 < state.end {
                usage.comparisons += 1;
                usage.step();
                if compare(&items[state.child], &items[state.child + 1], usage).is_lt() {
                    state.child += 1;
                }
            }
            usage.comparisons += 1;
            usage.step();
            if !compare(&items[state.root], &items[state.child], usage).is_lt() {
                break;
            }
            items.swap(state.root, state.child);
            usage.step();
            state.root = state.child;
        }
    }
    let mut state = GroupingSortState {
        root: items.len() / 2,
        child: 0,
        end: items.len(),
    };
    while state.root > 0 {
        let root = state.root - 1;
        state.root = root;
        sift(items, &mut state, usage, &mut compare);
        state.root = root;
    }
    while state.end > 1 {
        state.end -= 1;
        items.swap(0, state.end);
        usage.step();
        state.root = 0;
        sift(items, &mut state, usage, &mut compare);
    }
}

fn grouping_sort_comparisons(n: usize) -> Result<u128, RemoteGroupingError<'static>> {
    if n < 2 {
        return Ok(0);
    }
    let sifts = grouping_add((n / 2) as u128, (n - 1) as u128)?;
    grouping_mul(
        grouping_mul(2, sifts)?,
        (usize::BITS - n.leading_zeros()) as u128,
    )
}

impl<'a> RemoteGroupingQuote<'a> {
    /// Allocation-free O(operations + input references) metadata prepass. No key
    /// contents compared, opaque owner nodes priced, or legacy graph caps added.
    pub(crate) fn checked(graph: &'a DecoderGraph) -> Result<Self, RemoteGroupingError<'static>> {
        let n = graph.operations.len();
        let mut remote_count = 0;
        let mut max_id = 0u128;
        let mut key_units = 0u128;
        for op in &graph.operations {
            max_id = max_id.max(op.id.len() as u128);
            if remote_operator(op.operator) {
                remote_count += 1;
                let mut units = 4u128; // operator, layer, input length, singleton
                for id in &op.inputs {
                    units = grouping_add(units, grouping_add(id.len() as u128, 1)?)?;
                }
                if op.operator == ModelOperator::OutputHead {
                    units = grouping_add(units, grouping_add(op.id.len() as u128, 1)?)?;
                }
                key_units = key_units.max(units);
            }
        }
        let n128 = n as u128;
        let r128 = remote_count as u128;
        let c = grouping_sort_comparisons(n)?;
        let rc = grouping_sort_comparisons(remote_count)?;
        let comparison_bound = grouping_add(grouping_mul(c, 2)?, rc)?;
        let mut work_bound = 0;
        // Heap comparisons include their call unit; each sift swap is bounded
        // by a comparison, plus exactly len-1 extraction swaps per sort.
        for bound in [
            grouping_add(grouping_mul(c, 3)?, n.saturating_sub(1) as u128)?,
            grouping_add(
                grouping_mul(c, grouping_add(max_id, 4)?)?,
                n.saturating_sub(1) as u128,
            )?,
            grouping_add(
                grouping_mul(rc, grouping_add(key_units, 4)?)?,
                remote_count.saturating_sub(1) as u128,
            )?,
            n128,
            n128,
            n128, // row / ID / member initialization
            n128,
            n128,
            r128, // layers / weight-head scan / weight accessors
            grouping_mul(n128, grouping_add(max_id, 1)?)?, // ID-run boundaries
            n128,
            grouping_mul(r128, grouping_add(key_units, 1)?)?, // ID-run minimum remote key
            n128,
            r128, // ID metadata / duplicate error scan
            grouping_mul(r128, key_units)?,
            r128, // group boundaries / row fill
        ] {
            work_bound = grouping_add(work_bound, bound)?;
        }
        let heap = grouping_add(
            grouping_layout::<usize>(n)? as u128,
            grouping_add(
                grouping_layout::<GroupRow>(n)? as u128,
                grouping_layout::<RemoteMember<'_>>(remote_count)? as u128,
            )?,
        )?;
        let heap_bytes = usize::try_from(heap).map_err(|_| RemoteGroupingError::Overflow)?;
        // Concrete logical owner/scratch/result slots. Borrowed pointees,
        // native/dependency stacks, allocator overhead and RSS are excluded.
        let mut workspace_bytes = heap_bytes;
        for bytes in [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<RemoteGroups<'_>>(),
            std::mem::size_of::<GroupingBuildState>(),
            std::mem::size_of::<GroupingSortState>(),
            std::mem::size_of::<KeyCompareState>(),
            std::mem::size_of::<IndexAllocation<RemoteMember<'_>>>(),
            std::mem::size_of::<Result<Box<[RemoteMember<'_>]>, RemoteGroupingError<'_>>>(),
            std::mem::size_of::<Result<RemoteGroups<'_>, RemoteGroupingError<'_>>>(),
            std::mem::size_of::<Result<Self, RemoteGroupingError<'_>>>(),
            std::mem::size_of::<std::alloc::Layout>(),
            std::mem::size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>(),
            std::mem::size_of::<RemoteGroup<'_, '_>>(),
            std::mem::size_of::<RemoteMemberIter<'_, '_>>(),
            std::mem::size_of::<std::slice::Iter<'_, ModelOperation>>()
                .max(std::mem::size_of::<std::slice::Iter<'_, usize>>())
                .max(std::mem::size_of::<std::slice::Iter<'_, RemoteMember<'_>>>()),
            std::mem::size_of::<std::slice::Iter<'_, String>>(),
        ] {
            workspace_bytes = workspace_bytes
                .checked_add(bytes)
                .ok_or(RemoteGroupingError::Overflow)?;
        }
        Ok(Self {
            graph,
            remote_count,
            resources: RemoteGroupingResources {
                workspace_bytes,
                heap_bytes,
                work_bound,
                comparison_bound,
            },
        })
    }
    pub(crate) fn resources(&self) -> RemoteGroupingResources {
        self.resources
    }

    /// Caller-held original reservation is required when used for admission;
    /// these numerical ceilings do NOT create a Budget guard. Legacy callers
    /// use the identical exact allocation policy without claiming authority.
    pub(crate) fn construct(
        self,
        workspace_capacity: usize,
        max_work: u128,
    ) -> Result<RemoteGroups<'a>, RemoteGroupingError<'a>> {
        if workspace_capacity < self.resources.workspace_bytes {
            return Err(RemoteGroupingError::Capacity);
        }
        if max_work < self.resources.work_bound {
            return Err(RemoteGroupingError::WorkLimit);
        }
        let graph = self.graph;
        let n = graph.operations.len();
        let mut state = GroupingBuildState::default();
        let mut rows = grouping_box(n, |index| {
            state.usage.step();
            GroupRow {
                last_id_position: index,
                ..GroupRow::default()
            }
        })?;
        let mut by_id = grouping_box(n, |index| {
            state.usage.step();
            index
        })?;
        grouping_sort(&mut by_id, &mut state.usage, |a, b, usage| {
            usage.step();
            graph.operations[*a].layer.cmp(&graph.operations[*b].layer)
        });
        for &index in by_id.iter() {
            state.usage.step();
            if let Some(layer) = graph.operations[index].layer {
                if state.previous_layer != Some(layer) {
                    if layer != state.layers {
                        return Err(RemoteGroupingError::Layers);
                    }
                    state.layers = state
                        .layers
                        .checked_add(1)
                        .ok_or(RemoteGroupingError::Layers)?;
                    state.previous_layer = Some(layer);
                }
            }
        }
        if state.layers == 0 {
            return Err(RemoteGroupingError::Layers);
        }
        for op in &graph.operations {
            state.usage.step();
            if remote_operator(op.operator) {
                state.usage.step();
                if op
                    .attributes
                    .get("weight")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    return Err(RemoteGroupingError::Weight(&op.id));
                }
            }
            state.token_lookups += usize::from(op.operator == ModelOperator::TokenLookup);
            state.output_heads += usize::from(op.operator == ModelOperator::OutputHead);
        }
        if state.token_lookups == 0 || state.output_heads != 1 {
            return Err(RemoteGroupingError::LookupHead);
        }
        grouping_sort(&mut by_id, &mut state.usage, |a, b, usage| {
            let order =
                grouping_bytes_cmp(&graph.operations[*a].id, &graph.operations[*b].id, usage);
            if !order.is_eq() {
                return order;
            }
            usage.step();
            a.cmp(b)
        });
        while state.start < n {
            state.end = state.start + 1;
            while state.end < n
                && grouping_bytes_cmp(
                    &graph.operations[by_id[state.start]].id,
                    &graph.operations[by_id[state.end]].id,
                    &mut state.usage,
                )
                .is_eq()
            {
                state.end += 1;
            }
            state.first_remote = None;
            for &index in &by_id[state.start..state.end] {
                state.usage.step();
                if remote_operator(graph.operations[index].operator) {
                    let earlier = state.first_remote.is_none_or(|first| {
                        let order = remote_key_cmp(
                            &graph.operations[index],
                            &graph.operations[first],
                            &mut state.usage,
                        );
                        if !order.is_eq() {
                            order.is_lt()
                        } else {
                            state.usage.step();
                            index < first
                        }
                    });
                    if earlier {
                        state.first_remote = Some(index);
                    }
                }
            }
            for &index in &by_id[state.start..state.end] {
                state.usage.step();
                rows[index].last_id_position = by_id[state.end - 1];
                rows[index].remote_id_position = state.first_remote;
                rows[index].repeated_remote_id = remote_operator(graph.operations[index].operator)
                    && state.first_remote != Some(index);
            }
            state.start = state.end;
        }
        let mut members = grouping_box(self.remote_count, |_| {
            while !remote_operator(graph.operations[state.cursor].operator) {
                state.usage.step();
                state.cursor += 1;
            }
            state.usage.step();
            let index = state.cursor;
            state.cursor += 1;
            RemoteMember {
                index,
                operation: &graph.operations[index],
            }
        })?;
        grouping_sort(&mut members, &mut state.usage, |a, b, usage| {
            let order = remote_key_cmp(a.operation, b.operation, usage);
            if !order.is_eq() {
                return order;
            }
            usage.step();
            let order = rows[a.index]
                .last_id_position
                .cmp(&rows[b.index].last_id_position);
            if !order.is_eq() {
                return order;
            }
            usage.step();
            a.index.cmp(&b.index)
        });
        for member in members.iter() {
            state.usage.step();
            if rows[member.index].repeated_remote_id {
                return Err(RemoteGroupingError::Duplicate(&member.operation.id));
            }
        }
        state.start = 0;
        while state.start < members.len() {
            state.end = state.start + 1;
            while state.end < members.len()
                && remote_key_cmp(
                    members[state.start].operation,
                    members[state.end].operation,
                    &mut state.usage,
                )
                .is_eq()
            {
                state.end += 1;
            }
            for member in &members[state.start..state.end] {
                state.usage.step();
                rows[member.index].start = state.start;
                rows[member.index].len = state.end - state.start;
                rows[member.index].leader = member.index == members[state.start].index;
            }
            state.groups += 1;
            state.start = state.end;
        }
        if state.usage.work > self.resources.work_bound
            || state.usage.comparisons > self.resources.comparison_bound
        {
            return Err(RemoteGroupingError::WorkLimit);
        }
        Ok(RemoteGroups {
            graph,
            by_id,
            rows,
            members,
            groups: state.groups,
            #[cfg(test)]
            usage: state.usage,
        })
    }
}

pub(crate) fn checked_remote_groups(
    graph: &DecoderGraph,
) -> Result<RemoteGroups<'_>, RemoteGroupingError<'_>> {
    let quote = RemoteGroupingQuote::checked(graph)?;
    let resources = quote.resources();
    quote.construct(resources.workspace_bytes, resources.work_bound)
}

pub(crate) fn classify_remote_groups(graph: &DecoderGraph) -> Result<RemoteGroups<'_>, String> {
    checked_remote_groups(graph).map_err(|error| error.to_string())
}

fn graph_reaches(graph: &DecoderGraph, start: &str, target: ModelOperator, reverse: bool) -> bool {
    let mut frontier = vec![start];
    let mut visited = BTreeSet::new();
    for _ in 0..5 {
        let mut next = Vec::new();
        for id in frontier {
            if !visited.insert(id) {
                continue;
            }
            let Some(operation) = graph.operations.iter().find(|op| op.id == id) else {
                continue;
            };
            if operation.operator == target {
                return true;
            }
            if reverse
                && id != start
                && matches!(
                    operation.operator,
                    ModelOperator::Linear | ModelOperator::TokenLookup | ModelOperator::OutputHead
                )
            {
                continue;
            }
            if reverse {
                next.extend(operation.inputs.iter().map(String::as_str));
            } else {
                next.extend(
                    graph
                        .operations
                        .iter()
                        .filter(|op| op.inputs.iter().any(|input| input == id))
                        .map(|op| op.id.as_str()),
                );
            }
        }
        frontier = next;
    }
    false
}

fn linear_role(graph: &DecoderGraph, group: &RemoteGroup<'_, '_>) -> Option<&'static str> {
    let ops = &group.operations;
    if !ops.iter().all(|op| op.operator == ModelOperator::Linear) {
        return None;
    }
    if ops.len() == 3
        && ops
            .iter()
            .filter(|op| graph_reaches(graph, &op.id, ModelOperator::RotaryEmbedding, false))
            .count()
            == 2
    {
        return Some("qkv_projection");
    }
    if ops.len() == 2
        && ops
            .iter()
            .all(|op| graph_reaches(graph, &op.id, ModelOperator::Multiply, false))
        && ops
            .iter()
            .filter(|op| {
                graph_reaches(graph, &op.id, ModelOperator::Silu, false)
                    || graph_reaches(graph, &op.id, ModelOperator::GeluTanh, false)
            })
            .count()
            == 1
    {
        return Some("mlp_gate_up");
    }
    if ops.len() != 1 {
        return None;
    }
    let source = ops.first().inputs.first()?;
    if graph_reaches(graph, source, ModelOperator::AttentionValues, true) {
        return Some("attention_output");
    }
    let producer = graph.operations.iter().find(|op| op.id == *source)?;
    if producer.operator == ModelOperator::Multiply && producer.inputs.len() == 2 {
        let inputs: Vec<_> = producer
            .inputs
            .iter()
            .filter_map(|id| graph.operations.iter().find(|op| op.id == *id))
            .collect();
        if inputs
            .iter()
            .filter(|op| matches!(op.operator, ModelOperator::Silu | ModelOperator::GeluTanh))
            .count()
            == 1
            && inputs
                .iter()
                .filter(|op| op.operator == ModelOperator::Linear)
                .count()
                == 1
        {
            return Some("mlp_down");
        }
    }
    None
}

pub(crate) fn lower_phase(
    graph: &DecoderGraph,
    linear_executor: DecoderRuntimeExecutor,
    client_prefix_layers: u64,
    client_linear_roles: &BTreeSet<&str>,
    client_linear_layer_roles: &BTreeSet<(u64, &str)>,
) -> Result<DecoderRuntimePhaseSchedule, String> {
    let leaders = classify_remote_groups(graph)?;
    let mut selected_roles = BTreeSet::new();
    let mut selected_layer_roles = BTreeSet::new();
    let mut scheduled = BTreeSet::new();
    let mut steps = Vec::new();
    for (operation_index, operation) in graph.operations.iter().enumerate() {
        if leaders.is_remote(operation_index) {
            let Some(group) = leaders.at(operation_index) else {
                continue;
            };
            let input_ids = group.operations.first().inputs.clone();
            let layer = group.operations.first().layer;
            let role = linear_role(graph, &group);
            let selected = role.filter(|role| client_linear_roles.contains(role));
            if let Some(role) = selected {
                selected_roles.insert(role);
            }
            let selected_layer = layer
                .zip(role)
                .filter(|pair| client_linear_layer_roles.contains(pair));
            if let Some(pair) = selected_layer {
                selected_layer_roles.insert(pair);
            }
            if group
                .operations
                .iter()
                .any(|member| member.inputs != input_ids || member.layer != layer)
            {
                return Err("runtime stage operations do not share inputs and layer".into());
            }
            let mut offset = 0_u64;
            let mut outputs = Vec::new();
            let mut weight_ids = Vec::new();
            for member in group.operations.iter() {
                let width = last_width(member)?;
                outputs.push(DecoderRuntimeOutput {
                    operation_id: member.id.clone(),
                    output_shape: member.output_shape.clone(),
                    stage_offset: offset,
                    stage_width: width,
                });
                offset = offset
                    .checked_add(width)
                    .ok_or("runtime stage width overflowed")?;
                weight_ids.push(weight_id(member)?);
                if !scheduled.insert(member.id.as_str()) {
                    return Err(format!("operation {} was scheduled twice", member.id));
                }
            }
            steps.push(DecoderRuntimeStep {
                order: u64::try_from(steps.len()).map_err(|_| "runtime step count exceeds u64")?,
                operation_ids: group
                    .operations
                    .iter()
                    .map(|member| member.id.clone())
                    .collect(),
                operators: group
                    .operations
                    .iter()
                    .map(|member| member.operator)
                    .collect(),
                layer,
                input_ids,
                executor: if layer.is_some_and(|index| index < client_prefix_layers)
                    || selected.is_some()
                    || selected_layer.is_some()
                {
                    DecoderRuntimeExecutor::ClientLinear
                } else {
                    linear_executor
                },
                weight_ids,
                outputs,
                terminal_row_only: false,
            });
            continue;
        }
        if !local_operator(operation) {
            return Err(format!(
                "operation {} ({:?}) has no decoder runtime executor",
                operation.id, operation.operator
            ));
        }
        if operation.operator == ModelOperator::RotaryEmbedding
            && operation
                .attributes
                .get("frequency_scaling")
                .and_then(|scale| scale.get("kind"))
                .and_then(serde_json::Value::as_str)
                == Some("per_frequency_context")
        {
            let original = operation.attributes["frequency_scaling"]
                ["original_max_position_embeddings"]
                .as_u64()
                .ok_or("per-frequency rotary context is not bound")?;
            if graph.maximum_key_sequence > original {
                return Err("per-frequency rotary crossing requires cache re-rotation".into());
            }
        }
        if !scheduled.insert(operation.id.as_str()) {
            return Err(format!("operation {} was scheduled twice", operation.id));
        }
        steps.push(DecoderRuntimeStep {
            order: u64::try_from(steps.len()).map_err(|_| "runtime step count exceeds u64")?,
            operation_ids: vec![operation.id.clone()],
            operators: vec![operation.operator],
            layer: operation.layer,
            input_ids: operation.inputs.clone(),
            executor: DecoderRuntimeExecutor::ClientLocal,
            weight_ids: Vec::new(),
            terminal_row_only: false,
            outputs: vec![DecoderRuntimeOutput {
                operation_id: operation.id.clone(),
                output_shape: operation.output_shape.clone(),
                stage_offset: 0,
                stage_width: last_width(operation)?,
            }],
        });
    }
    let expected: BTreeSet<_> = graph
        .operations
        .iter()
        .map(|operation| operation.id.as_str())
        .collect();
    if scheduled != expected {
        return Err("decoder runtime schedule does not cover every operation".into());
    }
    if selected_roles != *client_linear_roles || selected_layer_roles != *client_linear_layer_roles
    {
        return Err("client linear role is absent or has no executable semantic stage".into());
    }
    if (!client_linear_roles.is_empty() || !client_linear_layer_roles.is_empty())
        && !steps.iter().any(|step| {
            step.layer.is_some()
                && matches!(
                    step.executor,
                    DecoderRuntimeExecutor::RemoteStage
                        | DecoderRuntimeExecutor::VerifiedRemoteStage
                )
        })
    {
        return Err("client linear placement must retain a remote body stage".into());
    }
    Ok(DecoderRuntimePhaseSchedule {
        mode: graph.mode,
        batch: graph.batch,
        query_sequence: graph.query_sequence,
        maximum_key_sequence: graph.maximum_key_sequence,
        state_inputs: graph.state_inputs.clone(),
        state_outputs: graph.state_outputs.clone(),
        steps,
        output: graph.output.clone(),
    })
}

type LinearStageSignature = (Option<u64>, Vec<String>, Vec<String>, Vec<u64>);

fn linear_stage_signature(phase: &DecoderRuntimePhaseSchedule) -> Vec<LinearStageSignature> {
    phase
        .steps
        .iter()
        .filter(|step| step.executor != DecoderRuntimeExecutor::ClientLocal)
        .map(|step| {
            (
                step.layer,
                step.operation_ids.clone(),
                step.weight_ids.clone(),
                step.outputs
                    .iter()
                    .map(|output| output.stage_width)
                    .collect(),
            )
        })
        .collect()
}

fn validate_window_resources(graph: &DecoderGraph) -> Result<(), String> {
    if !graph.operations.iter().any(|operation| {
        operation.operator == ModelOperator::CacheSuffix
            && operation
                .attributes
                .get("axis")
                .and_then(serde_json::Value::as_u64)
                == Some(3)
    }) {
        return Ok(());
    }
    let mut state_bytes = 0_u64;
    for state in &graph.state_outputs {
        let bytes = state
            .shape
            .iter()
            .try_fold(4_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or("bounded decoder state size overflowed")?;
        state_bytes = state_bytes
            .checked_add(bytes)
            .ok_or("bounded decoder state size overflowed")?;
        if state_bytes > MAX_WINDOW_STATE_BYTES {
            return Err("bounded decoder state exceeds the client memory ceiling".into());
        }
    }
    for operation in &graph.operations {
        if operation.operator != ModelOperator::KvCacheAppend || operation.output_shape.len() != 5 {
            continue;
        }
        let shape = &operation.output_shape;
        let view_elements = [shape[0], shape[1], shape[3], shape[4]]
            .into_iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(dimension))
            .ok_or("bounded window view size overflowed")?;
        if view_elements > MAX_WINDOW_VIEW_ELEMENTS {
            return Err("bounded window view exceeds the client memory ceiling".into());
        }
    }
    Ok(())
}

pub fn lower_decoder_runtime_schedule(
    plan: &DecoderPlan,
    canonical_composition: &[u8],
) -> Result<DecoderRuntimeSchedule, String> {
    let composition: super::ExperimentPipeline = serde_json::from_slice(canonical_composition)
        .map_err(|error| format!("invalid pipeline composition: {error}"))?;
    let linear_executor = match super::classify_decoder_composition(canonical_composition)? {
        super::DecoderCompositionKind::MaskedLinear => DecoderRuntimeExecutor::RemoteStage,
        super::DecoderCompositionKind::TwoOnlineOffsetLinear => DecoderRuntimeExecutor::RemoteStage,
        super::DecoderCompositionKind::ClientOnlyLinear => DecoderRuntimeExecutor::ClientLinear,
        // The schedule declares a distinct executor; it cannot itself prove that
        // the online session supplied the authenticated one-use verifier material.
        super::DecoderCompositionKind::VerifiedMaskedLinear => {
            DecoderRuntimeExecutor::VerifiedRemoteStage
        }
        super::DecoderCompositionKind::Other => {
            return Err(
                "decoder runtime schedule requires an admitted linear component composition".into(),
            );
        }
    };
    let client_prefix_layers = composition
        .components
        .get("placement")
        .and_then(|component| {
            component
                .params
                .get("layers")
                .or_else(|| component.params.get("prefix_layers"))
        })
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let client_linear_roles: BTreeSet<_> = composition
        .components
        .get("placement")
        .and_then(|component| component.params.get("roles"))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .collect();
    let client_linear_layer_roles: BTreeSet<_> = composition
        .components
        .get("placement")
        .and_then(|component| component.params.get("layer_roles"))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(super::linear_layer_role_pair)
        .collect();
    plan.validate().map_err(|error| error.to_string())?;
    if composition
        .components
        .get("quantization")
        .and_then(|component| component.params.get("causal_reduction"))
        .is_some()
    {
        for graph in [&plan.prefill, &plan.decode] {
            if graph
                .state_outputs
                .iter()
                .any(|state| !matches!(state.kind, StateKind::Key | StateKind::Value))
            {
                return Err("canonical prefix reduction requires full key/value state".into());
            }
            for operation in &graph.operations {
                if operation
                    .attributes
                    .get("output_dtype")
                    .is_some_and(|dtype| {
                        !matches!(dtype.as_str(), Some("float32" | "model_native"))
                    })
                    || operation.attributes.get("key_layout").is_some()
                    || operation.attributes.get("value_layout").is_some()
                    || (operation.operator == ModelOperator::CausalMask
                        && operation
                            .attributes
                            .as_object()
                            .is_none_or(|attrs| !attrs.is_empty()))
                {
                    return Err(
                        "canonical prefix reduction requires dense full causal float32 operations"
                            .into(),
                    );
                }
            }
        }
    }
    if client_prefix_layers > 0
        && !plan
            .prefill
            .operations
            .iter()
            .filter_map(|operation| operation.layer)
            .any(|index| index == client_prefix_layers)
    {
        return Err(
            "client-owned prefix requires at least one complete remote suffix layer".into(),
        );
    }
    if !plan.prefill.state_inputs.is_empty() {
        return Err("prefill declares persistent state without an initialization step".into());
    }
    bind_bfloat16_graph(&plan.prefill)?;
    bind_bfloat16_graph(&plan.decode)?;
    if !plan.transformations.is_empty() {
        return Err("baseline decoder runtime schedule does not support transformed plans".into());
    }
    if plan.prefill.batch != 1 || plan.decode.batch != 1 {
        return Err("baseline decoder runtime schedule requires batch one".into());
    }
    validate_window_resources(&plan.prefill)?;
    validate_window_resources(&plan.decode)?;
    validate_causal_convolution_contract(&plan.prefill)?;
    validate_causal_convolution_contract(&plan.decode)?;
    validate_gated_delta_contract(&plan.prefill)?;
    validate_gated_delta_contract(&plan.decode)?;
    validate_causal_convolution_handoff(plan)?;
    let mut prefill = lower_phase(
        &plan.prefill,
        linear_executor,
        client_prefix_layers,
        &client_linear_roles,
        &client_linear_layer_roles,
    )?;
    let mut decode = lower_phase(
        &plan.decode,
        linear_executor,
        client_prefix_layers,
        &client_linear_roles,
        &client_linear_layer_roles,
    )?;
    let nonlinear = composition
        .components
        .get("nonlinear")
        .map(crate::masked_wire_schedule::MaskedWireNumericContract::new)
        .transpose()?;
    if nonlinear.is_some() {
        crate::masked_wire_schedule::bind_regions(&plan.prefill, &mut prefill)?;
        crate::masked_wire_schedule::bind_regions(&plan.decode, &mut decode)?;
    }
    if composition
        .components
        .get("linear")
        .and_then(|component| component.params.get("prefill_pruning"))
        .and_then(serde_json::Value::as_str)
        == Some("terminal")
    {
        let demand = terminal_demand(&plan.prefill);
        for step in &mut prefill.steps {
            step.terminal_row_only = step.operators.iter().all(|op| *op == ModelOperator::Linear)
                && step
                    .operation_ids
                    .iter()
                    .all(|id| demand.get(id.as_str()) == Some(&1))
                && step.outputs.iter().all(|output| {
                    matches!(output.output_shape.as_slice(),
                    [1, rows, _] if *rows == plan.prefill.query_sequence)
                });
        }
    }
    if linear_stage_signature(&prefill) != linear_stage_signature(&decode) {
        return Err("prefill/decode linear stage contracts differ".into());
    }
    let mut schedule = DecoderRuntimeSchedule {
        schema_version: DECODER_RUNTIME_SCHEDULE_SCHEMA_VERSION.into(),
        composition_digest: pipeline_digest_bytes(canonical_composition),
        model_plan_digest: plan.digest(),
        model_config_digest: plan.config_digest.clone(),
        prefill,
        decode,
        protected_execution: false,
        complete: true,
        private_boundary: if crate::decoder_boundary::selected(&composition) {
            Some(crate::decoder_boundary::lower(plan, canonical_composition)?)
        } else {
            None
        },
        nonlinear,
        retained_state: None,
        mlp_tiling: None,
        decode_epochs: None,
        inflight_prefix: None,
        decode_reuse: None,
    };
    if let Some(inventory) = composition.components.get("inventory").filter(|value| {
        value.params.contains_key("decode_epoch_tokens")
            || value.params.contains_key("decode_epoch_schedule")
    }) {
        let mut epochs = serde_json::json!({
            "schema_version": "pllm.decode_epochs.v1",
            "composition_digest": schedule.composition_digest,
            "model_plan_digest": schedule.model_plan_digest,
            "maximum_key_sequence": plan.decode.maximum_key_sequence,
            "state_identity": "exact_compiled_generation_and_session",
            "material_lifetime": "fresh_one_use_inventory_per_epoch",
            "semantic_bounds": "unchanged_full_cap"
        });
        for field in ["decode_epoch_tokens", "decode_epoch_schedule"] {
            if let Some(value) = inventory.params.get(field) {
                epochs[field] = value.clone();
            }
        }
        schedule.decode_epochs = Some(epochs);
    }
    if let Some(cache) = composition.components.get("cache").filter(|value| {
        value
            .params
            .get("decode")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
    }) {
        schedule.decode_reuse = Some(serde_json::json!({
            "schema_version": "pllm.decode_reuse.v1",
            "composition_digest": schedule.composition_digest,
            "model_plan_digest": schedule.model_plan_digest,
            "maximum_key_sequence": plan.decode.maximum_key_sequence,
            "decode_max_bytes": cache.params["decode_max_bytes"],
            "client_additional_memory_bytes": cache.params["decode_max_bytes"],
            "ownership": "caller_owned_trusted_client_decode_cache",
            "resource_allowance": "additional_client_transient_and_metadata_envelope",
            "state_identity": "exact_source_composition_full_kv_logits_history_pending_rng",
            "material_admission": "lookup_before_fresh_one_token_epoch",
            "semantic_bounds": "unchanged_full_cap"
        }));
        let continuation = crate::decoder_continuation::lower_decoder_continuation_from_schedule(
            plan,
            canonical_composition,
            &schedule,
        )?;
        if !continuation.generated_prefix_canonical {
            return Err("decode reuse requires native canonical full-KV continuation".into());
        }
    }
    if let Some(cache) = composition.components.get("cache").filter(|value| {
        value
            .params
            .get("inflight")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
    }) {
        schedule.inflight_prefix = Some(serde_json::json!({
            "schema_version": "pllm.inflight_prefix.v1",
            "composition_digest": schedule.composition_digest,
            "model_plan_digest": schedule.model_plan_digest,
            "fixed_input_tokens": cache.params["fixed_input_tokens"],
            "inflight_max_requests": cache.params["inflight_max_requests"],
            "inflight_max_bytes": cache.params["inflight_max_bytes"],
            "ownership": "trusted_client_pending_prefill",
            "resource_allowance": "additional_to_completed_cache_and_per_response_state",
            "state_identity": "exact_compiled_generation_and_session",
            "traffic_accounting": "executed_stage_rows_only_not_noop_joins"
        }));
    }
    if let Some(policy) = composition
        .components
        .get("state")
        .and_then(|state| state.params.get("retention"))
    {
        let contract = crate::retained_decoder::build_contract(
            plan,
            &schedule,
            canonical_composition,
            policy.clone(),
            2 << 30,
        )?;
        schedule.retained_state =
            Some(serde_json::from_slice(&contract).map_err(|e| e.to_string())?);
    }
    if let Some(rows) = crate::mlp_tiling::selected_rows(&composition)? {
        let contract = crate::mlp_tiling::build_contract(plan, &schedule, rows)?;
        schedule.mlp_tiling = Some(serde_json::from_slice(&contract).map_err(|e| e.to_string())?);
    }
    Ok(schedule)
}

#[cfg(test)]
mod numeric_contract_tests {
    use super::*;
    use crate::decoder_matrix_inventory::{
        lower_decoder_matrix_inventory, DECODER_MATRIX_INVENTORY_SCHEMA_VERSION,
    };
    use serde_json::{json, Value};
    #[test]
    fn grouping_workspace_layout_and_work_arithmetic_fail_before_allocation() {
        GROUPING_ALLOCATIONS.with(|count| count.set(0));
        assert!(grouping_layout::<usize>(usize::MAX).is_err());
        assert!(grouping_add(u128::MAX, 1).is_err());
        assert!(grouping_mul(u128::MAX, 2).is_err());
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
    }
    use pllm_models::{lower_model_json, DecoderWorkload};
    use pllm_types::canonical_bytes;

    #[test]
    fn bounded_text_only_hybrid_decoder_builds_a_semantic_schedule() {
        let plan = lower_model_json(
            include_bytes!("../../pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 5,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        for graph in [&plan.prefill, &plan.decode] {
            validate_causal_convolution_contract(graph).unwrap();
            validate_gated_delta_contract(graph).unwrap();
            let recurrence = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::GatedDeltaRule)
                .unwrap();
            let retained = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::GatedDeltaStateUpdate)
                .unwrap();
            assert!(local_operator(recurrence));
            assert!(local_operator(retained));
            let mut wrong_scale = recurrence.clone();
            wrong_scale.attributes["query_scale"]["sqrt_denominator"] = serde_json::json!(256);
            assert!(local_operator(&wrong_scale));
            let mut forged_scale = graph.clone();
            forged_scale
                .operations
                .iter_mut()
                .find(|row| row.id == recurrence.id)
                .unwrap()
                .attributes["query_scale"]["sqrt_denominator"] = serde_json::json!(256);
            assert!(validate_gated_delta_contract(&forged_scale).is_err());
            let mut incorrect_pair = graph.clone();
            incorrect_pair
                .operations
                .iter_mut()
                .find(|row| row.id == retained.id)
                .unwrap()
                .inputs[5] = "foreign.state".into();
            assert!(validate_gated_delta_contract(&incorrect_pair).is_err());
            let mut unbound_decay = graph.clone();
            unbound_decay
                .operations
                .iter_mut()
                .find(|row| row.operator == ModelOperator::GatedDeltaDecay)
                .unwrap()
                .attributes["formula"] = serde_json::json!("unknown");
            assert!(validate_gated_delta_contract(&unbound_decay).is_err());
            let convolution = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::CausalConvolution)
                .unwrap();
            let update = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::ConvolutionStateUpdate)
                .unwrap();
            assert!(local_operator(convolution));
            assert!(local_operator(update));
            let mut incorrect_mode = convolution.clone();
            incorrect_mode.attributes["mode"] = serde_json::json!("unbound");
            assert!(!local_operator(&incorrect_mode));
            let mut incorrect_shape = update.clone();
            incorrect_shape.output_shape[1] += 1;
            assert!(!local_operator(&incorrect_shape));
            let mut incorrect_pair = graph.clone();
            let changed = incorrect_pair
                .operations
                .iter_mut()
                .find(|row| row.id == update.id)
                .unwrap();
            changed.attributes["kernel_size"] = serde_json::json!(3);
            assert!(validate_causal_convolution_contract(&incorrect_pair).is_err());
            let mut forged = graph.clone();
            let changed = forged
                .operations
                .iter_mut()
                .find(|row| row.id == convolution.id)
                .unwrap();
            changed.inputs.clear();
            assert!(validate_causal_convolution_contract(&forged).is_err());
            let partial_rotary = graph
                .operations
                .iter()
                .find(|row| row.operator == ModelOperator::RotaryEmbedding)
                .unwrap();
            assert!(local_operator(partial_rotary));
            let mut external_axes = partial_rotary.clone();
            external_axes.attributes["position_policy"] =
                serde_json::json!("three_independent_axes");
            assert!(!local_operator(&external_axes));
            let mut different_section = partial_rotary.clone();
            different_section.attributes["mrope_section"] = serde_json::json!([11, 11, 11]);
            assert!(!local_operator(&different_section));
        }
        validate_causal_convolution_handoff(&plan).unwrap();
        let mut wrong_handoff = plan.clone();
        wrong_handoff.decode.state_inputs[0].shape[1] += 1;
        assert!(validate_causal_convolution_handoff(&wrong_handoff).is_err());
        let mut wrong_kernel = plan.clone();
        wrong_kernel
            .decode
            .operations
            .iter_mut()
            .find(|row| row.operator == ModelOperator::CausalConvolution)
            .unwrap()
            .attributes["weight"] = serde_json::json!("forged.other_weight");
        assert!(validate_causal_convolution_handoff(&wrong_kernel).is_err());
        let mut wrong_recurrent = plan.clone();
        wrong_recurrent
            .decode
            .state_inputs
            .iter_mut()
            .find(|row| row.kind == StateKind::Recurrent)
            .unwrap()
            .shape[2] += 1;
        assert!(validate_causal_convolution_handoff(&wrong_recurrent).is_err());
        let mut wrong_decay_weight = plan.clone();
        wrong_decay_weight
            .decode
            .operations
            .iter_mut()
            .find(|row| row.operator == ModelOperator::GatedDeltaDecay)
            .unwrap()
            .attributes["a_log"] = serde_json::json!("foreign.coefficient");
        assert!(validate_causal_convolution_handoff(&wrong_decay_weight).is_err());
        let composition = canonical_bytes(&serde_json::json!({
            "components": {
                "inference": {"component": "pllm/inference", "params": {}},
                "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
                "linear": {"component": "pllm/masked-linear", "params": {}},
                "preparation": {"component": "pllm/model-aware-corrections", "params": {}}
            },
            "model": {"source": "model.fixture"}
        }));
        let schedule = lower_decoder_runtime_schedule(&plan, &composition).unwrap();
        assert_eq!(schedule.prefill.state_inputs.len(), 0);
        assert_eq!(
            schedule.prefill.state_outputs.len(),
            schedule.decode.state_inputs.len()
        );
        assert_eq!(schedule.prefill.state_outputs.len(), 64);
        assert!(schedule.prefill.steps.iter().any(|step| step
            .operation_ids
            .iter()
            .any(|id| id.ends_with("gated_delta_rule"))));
        let state_operation = plan
            .decode
            .operations
            .iter()
            .find(|row| row.operator == ModelOperator::ConvolutionStateUpdate)
            .unwrap();
        assert!(schedule
            .decode
            .steps
            .iter()
            .any(|step| step.operation_ids.contains(&state_operation.id)));
    }

    #[test]
    fn checked_semantic_numeric_operations_require_bound_contracts() {
        let plan = lower_model_json(
            include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 2,
                max_new_tokens: 2,
            },
        )
        .unwrap();
        let groups = classify_remote_groups(&plan.prefill).unwrap();
        let group = groups.get("main_embedding").unwrap();
        assert_eq!(group.operations.len(), 2);
        assert_ne!(
            weight_id(group.operations.first()).unwrap(),
            weight_id(group.operations.iter().nth(1).unwrap()).unwrap()
        );
        bind_bfloat16_graph(&plan.prefill).unwrap();
        bind_bfloat16_graph(&plan.decode).unwrap();
        let mut unbound_linear = plan.prefill.clone();
        unbound_linear
            .operations
            .iter_mut()
            .find(|row| row.id == "layer.0.q_linear")
            .unwrap()
            .attributes["output_dtype"] = serde_json::Value::Null;
        assert!(bind_bfloat16_graph(&unbound_linear)
            .unwrap_err()
            .contains("layer.0.q_linear"));
        let mut unbound_residual = plan.prefill.clone();
        unbound_residual
            .operations
            .iter_mut()
            .find(|row| row.id == "layer.0.attention_residual")
            .unwrap()
            .attributes["output_dtype"] = serde_json::Value::Null;
        assert!(bind_bfloat16_graph(&unbound_residual)
            .unwrap_err()
            .contains("attention_residual"));
        let mut stripped = plan.prefill.clone();
        for operation in &mut stripped.operations {
            if let Some(attributes) = operation.attributes.as_object_mut() {
                attributes.remove("output_dtype");
            }
        }
        assert!(bind_bfloat16_graph(&stripped)
            .unwrap_err()
            .contains("main_embedding"));
        let operations = &plan.prefill.operations;
        for id in [
            "main_embedding_scaled",
            "ple_context_scaled",
            "layer.0.attention_scale",
            "layer.0.gelu_tanh",
            "layer.0.q_permute",
            "layer.0.ple_slice",
            "logit_softcap",
        ] {
            let operation = operations.iter().find(|row| row.id == id).unwrap();
            assert!(
                local_operator(operation),
                "{id} needs a generic client-local contract"
            );
        }
        let scalar = operations
            .iter()
            .find(|row| row.id == "layer.0.layer_scalar")
            .unwrap();
        assert!(local_operator(scalar));
        let mut forged_scalar = scalar.clone();
        forged_scalar.attributes["factor"]["weight_shape"] = serde_json::json!([2]);
        assert!(!local_operator(&forged_scalar));
        let suffix = operations
            .iter()
            .find(|row| row.id == "layer.0.key_suffix")
            .unwrap();
        assert!(local_operator(suffix));
        let mut forged_suffix = suffix.clone();
        forged_suffix.attributes["maximum_sequence"] = serde_json::json!(0);
        assert!(!local_operator(&forged_suffix));
        validate_window_resources(&plan.prefill).unwrap();
        validate_window_resources(&plan.decode).unwrap();
        let mut oversize = plan.decode.clone();
        oversize.state_outputs[0].shape[2] = 1 << 30;
        assert!(validate_window_resources(&oversize)
            .unwrap_err()
            .contains("memory ceiling"));
        let sliding_mask = operations
            .iter()
            .find(|row| row.id == "layer.0.causal_mask")
            .unwrap();
        assert!(local_operator(sliding_mask));
        let full_mask = operations
            .iter()
            .find(|row| {
                row.operator == ModelOperator::CausalMask && row.attributes["kind"] == "full_causal"
            })
            .unwrap();
        assert!(local_operator(full_mask));
        for id in ["layer.0.input_norm", "layer.0.v_norm", "layer.0.softmax"] {
            let operation = operations.iter().find(|row| row.id == id).unwrap();
            assert!(local_operator(operation));
        }
        let mut forged_norm = operations
            .iter()
            .find(|row| row.id == "layer.0.input_norm")
            .unwrap()
            .clone();
        forged_norm.attributes["with_scale"] = serde_json::json!(false);
        assert!(!local_operator(&forged_norm));
        for id in ["layer.0.attention_scores", "layer.0.attention_values"] {
            let operation = operations.iter().find(|row| row.id == id).unwrap();
            assert!(local_operator(operation));
            assert_eq!(operation.attributes["output_dtype"], "bfloat16");
        }
        let mut forged_layout = operations
            .iter()
            .find(|row| row.id == "layer.0.attention_scores")
            .unwrap()
            .clone();
        forged_layout.attributes["key_layout"] = serde_json::json!("unbound_layout");
        assert!(!local_operator(&forged_layout));
        let mut forged_window = sliding_mask.clone();
        forged_window.attributes["left_context"] = serde_json::json!(0);
        assert!(!local_operator(&forged_window));
        let mut forged = operations
            .iter()
            .find(|row| row.id == "main_embedding_scaled")
            .unwrap()
            .clone();
        forged.attributes["factor"]["factor_rounding_dtype"] = serde_json::json!("float16");
        assert!(!local_operator(&forged));
    }

    fn grouping_regression_plan(family: &str, tied: bool) -> DecoderPlan {
        let mut config = json!({
            "model_type": family, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "vocab_size": 32, "max_position_embeddings": 32, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": tied
        });
        if family == "qwen3" {
            for (key, value) in [
                ("head_dim", json!(4)),
                ("attention_bias", json!(false)),
                ("attention_dropout", json!(0.0)),
                ("rope_scaling", Value::Null),
                ("sliding_window", Value::Null),
                ("use_sliding_window", json!(false)),
                ("use_cache", json!(true)),
                ("max_window_layers", json!(2)),
                ("layer_types", json!(["full_attention", "full_attention"])),
            ] {
                config[key] = value;
            }
        }
        lower_model_json(
            &canonical_bytes(&config),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 3,
                max_new_tokens: 2,
            },
        )
        .unwrap()
    }

    // Independent reference for the old B-tree grouping, including original owned
    // keys, position sort and error precedence. Test-only, never an alternate path.
    fn legacy_groups(graph: &DecoderGraph) -> Result<BTreeMap<&str, Vec<&ModelOperation>>, String> {
        let layers: BTreeSet<_> = graph.operations.iter().filter_map(|op| op.layer).collect();
        if layers.is_empty() || layers.iter().copied().ne(0..layers.len() as u64) {
            return Err("decoder runtime layers must be contiguous from zero".into());
        }
        let mut grouped = BTreeMap::new();
        for op in &graph.operations {
            if remote_operator(op.operator) {
                if op
                    .attributes
                    .get("weight")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    return Err(format!(
                        "remote operation {} requires one declared weight artifact",
                        op.id
                    ));
                }
                grouped
                    .entry((
                        op.operator,
                        op.layer,
                        op.inputs.clone(),
                        (op.operator == ModelOperator::OutputHead).then(|| op.id.clone()),
                    ))
                    .or_insert_with(Vec::new)
                    .push(op);
            }
        }
        if !graph
            .operations
            .iter()
            .any(|op| op.operator == ModelOperator::TokenLookup)
            || graph
                .operations
                .iter()
                .filter(|op| op.operator == ModelOperator::OutputHead)
                .count()
                != 1
        {
            return Err("decoder runtime requires token lookup and one output head".into());
        }
        let positions: BTreeMap<_, _> = graph
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| (op.id.as_str(), i))
            .collect();
        let mut result = BTreeMap::new();
        let mut members = BTreeSet::new();
        for mut group in grouped.into_values() {
            group.sort_by_key(|op| positions[op.id.as_str()]);
            for op in &group {
                if !members.insert(op.id.as_str()) {
                    return Err(format!(
                        "operation {} belongs to multiple runtime stages",
                        op.id
                    ));
                }
            }
            let leader = group[0].id.as_str();
            assert!(result.insert(leader, group).is_none());
        }
        Ok(result)
    }

    // Valid-input oracle, independent of new borrowed structs, lowering and stream
    // emitter. Recreates the original owned JSON schema and original B-tree order.
    fn legacy_inventory_json(model: &DecoderPlan) -> Value {
        let phase = |graph: &DecoderGraph| {
            let vocabulary = graph
                .operations
                .iter()
                .find(|op| op.operator == ModelOperator::OutputHead)
                .unwrap()
                .output_shape[1];
            let reference = legacy_groups(graph).unwrap();
            let mut groups = Vec::new();
            for op in &graph.operations {
                let Some(members) = reference.get(op.id.as_str()) else {
                    continue;
                };
                let token = op.operator == ModelOperator::TokenLookup;
                let input_shape = if token {
                    vec![graph.batch, graph.query_sequence]
                } else {
                    graph
                        .operations
                        .iter()
                        .find(|producer| producer.id == op.inputs[0])
                        .unwrap()
                        .output_shape
                        .clone()
                };
                let mut offset = 0;
                let members: Vec<_> = members.iter().map(|member| {
                    let width = *member.output_shape.last().unwrap();
                    let bias = member.attributes.get("bias").and_then(Value::as_str);
                    let value = json!({
                        "operation_id": member.id, "weight_id": member.attributes["weight"],
                        "bias_id": bias, "source_shape": if token { [vocabulary, width] } else { [width, *input_shape.last().unwrap()] },
                        "bias_shape": bias.map(|_| [width]), "output_shape": member.output_shape,
                        "output_offset": offset, "output_width": width,
                    });
                    offset += width;
                    value
                }).collect();
                groups.push(json!({
                    "order": groups.len(), "operator": op.operator, "layer": op.layer,
                    "input_id": op.inputs[0], "input_shape": input_shape,
                    "artifact_kind": if token { "token_table" } else { "matrix" },
                    "orientation": if token { "vocabulary_feature" } else { "output_input" },
                    "logical_shape": if token { [vocabulary, offset] } else { [offset, *input_shape.last().unwrap()] },
                    "members": members,
                }));
            }
            json!({ "mode": graph.mode, "batch": graph.batch, "query_sequence": graph.query_sequence,
                "maximum_key_sequence": graph.maximum_key_sequence, "groups": groups })
        };
        json!({ "schema_version": DECODER_MATRIX_INVENTORY_SCHEMA_VERSION,
            "model_plan_digest": model.digest(), "model_config_digest": model.config_digest,
            "prefill": phase(&model.prefill), "decode": phase(&model.decode) })
    }

    #[test]
    fn borrowed_groups_match_original_algorithm_and_original_owners() {
        for family in ["llama", "qwen2", "qwen3"] {
            let mut model = grouping_regression_plan(family, true);
            for graph in [&mut model.prefill, &mut model.decode] {
                let mut table = graph.operations[0].clone();
                table.id = "nonadjacent-table".into();
                graph.operations.insert(2, table);
                let expected = legacy_groups(graph).unwrap();
                let actual = checked_remote_groups(graph).unwrap();
                assert_eq!(actual.len(), expected.len());
                for (index, op) in graph.operations.iter().enumerate() {
                    assert_eq!(actual.is_remote(index), remote_operator(op.operator));
                    match (actual.get(&op.id), expected.get(op.id.as_str())) {
                        (Some(actual), Some(expected)) => {
                            assert_eq!(actual.operations.len(), expected.len());
                            for (a, b) in actual.operations.iter().zip(expected) {
                                assert!(std::ptr::eq(a, *b));
                            }
                        }
                        (None, None) => (),
                        _ => panic!("leader mismatch: {}", op.id),
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_grouping_preserves_original_error_winner_without_error_allocation() {
        for case in 0..8 {
            let mut model = grouping_regression_plan("qwen2", true);
            let graph = &mut model.prefill;
            match case {
                0 => graph.operations.iter_mut().for_each(|op| op.layer = None),
                1 => graph
                    .operations
                    .iter_mut()
                    .filter(|op| op.layer == Some(1))
                    .for_each(|op| op.layer = Some(3)),
                2 => graph.operations[0].attributes["weight"] = Value::Null,
                3 => graph
                    .operations
                    .retain(|op| op.operator != ModelOperator::TokenLookup),
                4 => {
                    let head = graph
                        .operations
                        .iter()
                        .find(|op| op.operator == ModelOperator::OutputHead)
                        .unwrap()
                        .clone();
                    graph.operations.push(head);
                }
                5..=7 => {
                    let indices: Vec<_> = graph
                        .operations
                        .iter()
                        .enumerate()
                        .filter(|(_, op)| op.operator == ModelOperator::Linear)
                        .map(|(i, _)| i)
                        .collect();
                    graph.operations[indices[1]].id = graph.operations[indices[0]].id.clone();
                    if case >= 6 {
                        graph.operations[indices[4]].id = graph.operations[indices[3]].id.clone();
                    }
                    if case == 7 {
                        graph.operations[indices[1]].inputs = vec!["different-input".into()];
                    }
                }
                _ => unreachable!(),
            }
            assert_eq!(
                checked_remote_groups(graph).unwrap_err().to_string(),
                legacy_groups(graph).unwrap_err()
            );
        }
    }

    fn grouping_scale_graph(count: usize, shared_key: bool) -> DecoderGraph {
        let model = grouping_regression_plan("llama", true);
        let mut graph = model.prefill;
        let table = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::TokenLookup)
            .unwrap()
            .clone();
        let linear = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::Linear)
            .unwrap()
            .clone();
        let head = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::OutputHead)
            .unwrap()
            .clone();
        graph.operations = vec![table];
        for index in 0..count {
            let mut op = linear.clone();
            op.id = format!("matrix.{index:08}");
            op.layer = Some(if shared_key { 0 } else { index as u64 });
            op.inputs = vec!["shared.semantic.input".into()];
            graph.operations.push(op);
        }
        graph.operations[1..].reverse();
        graph.operations.push(head);
        graph
    }

    #[test]
    fn indexed_grouping_work_and_accesses_scale_n_log_n_not_pairs() {
        for shared_key in [false, true] {
            let mut previous_bound = None;
            for count in [32, 64, 128, 256, 512, 1024, 2048] {
                let graph = grouping_scale_graph(count, shared_key);
                let quote = RemoteGroupingQuote::checked(&graph).unwrap();
                let resources = quote.resources();
                let groups = quote
                    .construct(resources.workspace_bytes, resources.work_bound)
                    .unwrap();
                assert_eq!(groups.retained_index_bytes(), resources.heap_bytes);
                assert!(groups.usage().work <= resources.work_bound);
                assert!(groups.usage().comparisons <= resources.comparison_bound);
                if count >= 128 {
                    assert!(groups.usage().comparisons < (count as u128) * (count as u128));
                }
                if let Some(previous) = previous_bound {
                    // IDs/input lengths fixed; doubling n grows the explicit
                    // heapsort + linear reserve by <3x, never a quadratic 4x.
                    assert!(resources.work_bound < 3 * previous);
                }
                previous_bound = Some(resources.work_bound);
                assert_eq!(groups.len(), if shared_key { 3 } else { count + 2 });
                GROUPING_ACCESS.with(|counts| counts.set(GroupingAccess::default()));
                for index in 0..graph.operations.len() {
                    if groups.is_remote(index) {
                        if let Some(group) = groups.at(index) {
                            let cached = group.operations.len();
                            assert_eq!(group.operations.iter().count(), cached);
                        }
                    }
                }
                GROUPING_ACCESS.with(|counts| {
                    let counts = counts.get();
                    assert_eq!(counts.positional_probes, graph.operations.len());
                    assert_eq!(counts.id_probes, 0);
                    assert_eq!(counts.member_visits, graph.operations.len());
                });
                GROUPING_ACCESS.with(|counts| counts.set(GroupingAccess::default()));
                for op in &graph.operations {
                    let _ = groups.get(&op.id);
                }
                GROUPING_ACCESS.with(|counts| {
                    assert!(
                        counts.get().id_probes
                            <= graph.operations.len()
                                * (usize::BITS - graph.operations.len().leading_zeros()) as usize
                    )
                });
            }
        }
    }

    #[test]
    fn indexed_grouping_workspace_and_work_one_short_precede_all_three_factories() {
        let graph = grouping_scale_graph(128, true);
        GROUPING_ALLOCATIONS.with(|count| count.set(0));
        let quote = RemoteGroupingQuote::checked(&graph).unwrap();
        let resources = quote.resources();
        assert!(matches!(
            quote.construct(resources.workspace_bytes - 1, resources.work_bound),
            Err(RemoteGroupingError::Capacity)
        ));
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
        let quote = RemoteGroupingQuote::checked(&graph).unwrap();
        assert!(matches!(
            quote.construct(resources.workspace_bytes, resources.work_bound - 1),
            Err(RemoteGroupingError::WorkLimit)
        ));
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 0));
        let groups = RemoteGroupingQuote::checked(&graph)
            .unwrap()
            .construct(resources.workspace_bytes, resources.work_bound)
            .unwrap();
        GROUPING_ALLOCATIONS.with(|count| assert_eq!(count.get(), 3));
        assert_eq!(groups.retained_index_bytes(), resources.heap_bytes);
        let expected = legacy_groups(&graph).unwrap();
        for op in &graph.operations {
            let actual = groups.get(&op.id).map(|group| {
                group
                    .operations
                    .iter()
                    .map(|op| op.id.as_str())
                    .collect::<Vec<_>>()
            });
            let expected = expected
                .get(op.id.as_str())
                .map(|members| members.iter().map(|op| op.id.as_str()).collect::<Vec<_>>());
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn indexed_grouping_preserves_shadowed_ids_empty_ids_and_error_precedence() {
        for case in 0..5 {
            let mut model = grouping_regression_plan("qwen2", true);
            let graph = &mut model.prefill;
            match case {
                0 => {
                    let mut local = graph
                        .operations
                        .iter()
                        .find(|op| !remote_operator(op.operator))
                        .unwrap()
                        .clone();
                    local.id = graph
                        .operations
                        .iter()
                        .find(|op| op.operator == ModelOperator::Linear)
                        .unwrap()
                        .id
                        .clone();
                    graph.operations.push(local);
                }
                1 => graph
                    .operations
                    .iter_mut()
                    .find(|op| op.operator == ModelOperator::Linear)
                    .unwrap()
                    .id
                    .clear(),
                2 => {
                    graph.operations[0].attributes["weight"] = Value::Null;
                    graph.operations.iter_mut().for_each(|op| op.layer = None);
                }
                3 => {
                    graph.operations[0].attributes["weight"] = Value::Null;
                    graph
                        .operations
                        .retain(|op| op.operator != ModelOperator::OutputHead);
                }
                4 => {
                    let indices: Vec<_> = graph
                        .operations
                        .iter()
                        .enumerate()
                        .filter(|(_, op)| op.operator == ModelOperator::Linear)
                        .map(|(index, _)| index)
                        .collect();
                    for &index in &indices {
                        graph.operations[index].id = "duplicate.remote".into();
                    }
                }
                _ => unreachable!(),
            }
            match (checked_remote_groups(graph), legacy_groups(graph)) {
                (Err(actual), Err(expected)) => assert_eq!(actual.to_string(), expected),
                (Ok(actual), Ok(expected)) => {
                    assert_eq!(actual.len(), expected.len());
                    for op in &graph.operations {
                        let actual = actual.get(&op.id).map(|group| {
                            group
                                .operations
                                .iter()
                                .map(|op| op.id.as_str())
                                .collect::<Vec<_>>()
                        });
                        let expected = expected.get(op.id.as_str()).map(|members| {
                            members.iter().map(|op| op.id.as_str()).collect::<Vec<_>>()
                        });
                        assert_eq!(actual, expected);
                    }
                }
                (actual, expected) => panic!("oracle mismatch: {actual:?} / {expected:?}"),
            }
        }
    }

    #[test]
    fn indexed_grouping_retains_existing_legacy_schedule_golden_digest() {
        // Same fixture/domain/composition and known digest as the pre-existing
        // delegation_publication_inventory legacy schedule regression; not rebaselined.
        let model = grouping_regression_plan("qwen2", true);
        let composition = canonical_bytes(&json!({
            "components": {
                "inference": {"component": "pllm/inference", "params": {}},
                "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
                "linear": {"component": "pllm/masked-linear", "params": {}},
                "preparation": {"component": "pllm/model-aware-corrections", "params": {}}
            },
            "model": {"source": "model.fixture"}
        }));
        GROUPING_ACCESS.with(|counts| counts.set(GroupingAccess::default()));
        let schedule =
            crate::decoder_runtime_schedule::lower_decoder_runtime_schedule(&model, &composition)
                .unwrap();
        GROUPING_ACCESS.with(|counts| assert_eq!(counts.get().id_probes, 0));
        assert_eq!(
            schedule.digest().as_str(),
            "9b2f3bed2dff2d83598bebff5cabf5a22f0ed30f470d8b7d35215d1aaa57719d"
        );
        assert_eq!(
            lower_decoder_matrix_inventory(&model)
                .unwrap()
                .canonical_bytes(),
            canonical_bytes(&legacy_inventory_json(&model))
        );
    }

    #[test]
    fn indexed_id_lookup_does_not_walk_duplicate_nonremote_id_runs() {
        let mut graph = grouping_scale_graph(32, false);
        let id = graph.operations[1].id.clone();
        let mut local = graph.operations[1].clone();
        local.operator = ModelOperator::RmsNorm;
        local.layer = None;
        graph.operations.extend((0..1024).map(|_| local.clone()));
        let groups = checked_remote_groups(&graph).unwrap();
        let original = legacy_groups(&graph).unwrap();
        GROUPING_ACCESS.with(|counts| counts.set(GroupingAccess::default()));
        for _ in 0..256 {
            let actual = groups.get(&id).unwrap();
            assert!(std::ptr::eq(
                actual.operations.first(),
                original[id.as_str()][0]
            ));
        }
        GROUPING_ACCESS.with(|counts| {
            assert!(
                counts.get().id_probes
                    <= 256 * (usize::BITS - graph.operations.len().leading_zeros()) as usize
            )
        });
    }

    // Original lower_phase's ID-membership dispatch and duplicate-error order,
    // independent of the indexed rows. Fixtures otherwise use valid operators.
    fn legacy_dispatch_ids(graph: &DecoderGraph) -> Result<Vec<Vec<&str>>, String> {
        let leaders = legacy_groups(graph)?;
        let remote_members: BTreeSet<_> = leaders
            .values()
            .flat_map(|group| group.iter().map(|op| op.id.as_str()))
            .collect();
        let mut scheduled = BTreeSet::new();
        let mut steps = Vec::new();
        for op in &graph.operations {
            let ids = if remote_members.contains(op.id.as_str()) {
                let Some(group) = leaders.get(op.id.as_str()) else {
                    continue;
                };
                group.iter().map(|op| op.id.as_str()).collect::<Vec<_>>()
            } else {
                vec![op.id.as_str()]
            };
            for id in &ids {
                if !scheduled.insert(*id) {
                    return Err(format!("operation {id} was scheduled twice"));
                }
            }
            steps.push(ids);
        }
        Ok(steps)
    }

    #[test]
    fn indexed_lower_phase_matches_legacy_id_dispatch_for_local_shadows() {
        let model = grouping_regression_plan("qwen2", true);
        for original in [&model.prefill, &model.decode] {
            let baseline = lower_phase(
                original,
                DecoderRuntimeExecutor::RemoteStage,
                0,
                &BTreeSet::new(),
                &BTreeSet::new(),
            )
            .unwrap();
            let reference = legacy_groups(original).unwrap();
            let multi = reference
                .values()
                .find(|group| group.len() > 1 && group[0].operator == ModelOperator::Linear)
                .unwrap();
            for leader in [false, true] {
                let target = if leader {
                    multi[0]
                } else {
                    multi[multi.len() - 1]
                };
                for after in [false, true] {
                    let mut graph = original.clone();
                    let mut local = graph
                        .operations
                        .iter()
                        .find(|op| op.operator == ModelOperator::RmsNorm)
                        .unwrap()
                        .clone();
                    local.id = target.id.clone();
                    assert!(local_operator(&local));
                    let remote_position = graph
                        .operations
                        .iter()
                        .position(|op| op.id == target.id)
                        .unwrap();
                    // A leader's adjacent shadow leaves it first in the remote
                    // group. Nonleader shadows exercise both graph extremes,
                    // including the reviewer's appended-local reproduction.
                    let position = if leader {
                        remote_position + usize::from(after)
                    } else if after {
                        graph.operations.len()
                    } else {
                        0
                    };
                    graph.operations.insert(position, local);
                    assert_eq!(
                        legacy_groups(&graph)
                            .unwrap()
                            .contains_key(target.id.as_str()),
                        leader
                    );
                    let expected = legacy_dispatch_ids(&graph);
                    GROUPING_ACCESS.with(|counts| counts.set(GroupingAccess::default()));
                    let actual = lower_phase(
                        &graph,
                        DecoderRuntimeExecutor::RemoteStage,
                        0,
                        &BTreeSet::new(),
                        &BTreeSet::new(),
                    );
                    GROUPING_ACCESS.with(|counts| assert_eq!(counts.get().id_probes, 0));
                    match (actual, expected) {
                        (Err(actual), Err(expected)) => {
                            assert!(leader);
                            assert_eq!(
                                expected,
                                format!("operation {} was scheduled twice", target.id)
                            );
                            assert_eq!(actual, expected);
                        }
                        (Ok(actual), Ok(expected)) => {
                            assert!(!leader);
                            let ids: Vec<Vec<_>> = actual
                                .steps
                                .iter()
                                .map(|step| step.operation_ids.iter().map(String::as_str).collect())
                                .collect();
                            assert_eq!(ids, expected);
                            assert_eq!(canonical_bytes(&actual), canonical_bytes(&baseline));
                        }
                        (actual, expected) => {
                            panic!("shadow dispatch mismatch: {actual:?} / {expected:?}")
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn indexed_grouping_orders_multi_input_prefix_empty_and_utf8_keys() {
        let mut graph = grouping_regression_plan("llama", true).prefill;
        let table = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::TokenLookup)
            .unwrap()
            .clone();
        let head = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::OutputHead)
            .unwrap()
            .clone();
        let linear = graph
            .operations
            .iter()
            .find(|op| op.operator == ModelOperator::Linear)
            .unwrap()
            .clone();
        let prefix = "shared-prefix-".repeat(128);
        let cases: Vec<Vec<String>> = vec![
            vec![],
            vec!["".into()],
            vec!["a".into()],
            vec!["a".into(), "".into()],
            vec!["a".into(), "b".into()],
            vec!["b".into(), "a".into()],
            vec!["a".into(), "b".into(), "".into()],
            vec!["ab".into()],
            vec!["ab".into(), "".into()],
            vec!["\0".into()],
            vec!["é".into(), "λ".into()],
            vec!["λ".into(), "é".into()],
            vec![format!("{prefix}a")],
            vec![format!("{prefix}b")],
            vec![format!("{prefix}a"), "é".into()],
            vec!["a".into(), "b".into()],
        ];
        graph.operations = vec![table];
        for (index, inputs) in cases.into_iter().enumerate() {
            let mut op = linear.clone();
            op.id = format!("key.{index}");
            op.layer = Some(0);
            op.inputs = inputs;
            graph.operations.push(op);
        }
        graph.operations[1..].reverse();
        graph.operations.push(head);
        for a in &graph.operations {
            for b in &graph.operations {
                let original_key = |op: &ModelOperation| {
                    (
                        op.operator,
                        op.layer,
                        op.inputs.clone(),
                        (op.operator == ModelOperator::OutputHead).then(|| op.id.clone()),
                    )
                };
                let mut usage = GroupingUsage::default();
                assert_eq!(
                    remote_key_cmp(a, b, &mut usage),
                    original_key(a).cmp(&original_key(b))
                );
            }
        }
        let expected = legacy_groups(&graph).unwrap();
        let quote = RemoteGroupingQuote::checked(&graph).unwrap();
        let resources = quote.resources();
        let actual = quote
            .construct(resources.workspace_bytes, resources.work_bound)
            .unwrap();
        assert!(actual.usage().work <= resources.work_bound);
        assert!(actual.usage().comparisons <= resources.comparison_bound);
        assert_eq!(actual.len(), expected.len());
        for op in &graph.operations {
            let actual = actual.get(&op.id).map(|group| {
                group
                    .operations
                    .iter()
                    .map(|member| member.id.as_str())
                    .collect::<Vec<_>>()
            });
            let expected = expected.get(op.id.as_str()).map(|group| {
                group
                    .iter()
                    .map(|member| member.id.as_str())
                    .collect::<Vec<_>>()
            });
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn indexed_grouping_heapsort_boundary_lengths_obey_meter_bounds() {
        for len in [
            0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
            255, 256, 257, 1023, 1024, 1025,
        ] {
            for pattern in 0..3 {
                let mut values: Vec<_> = (0..len)
                    .map(|i| match pattern {
                        0 => len - i,
                        1 => 0,
                        _ => (i * 37) % len.max(1),
                    })
                    .collect();
                let mut expected = values.clone();
                expected.sort_unstable();
                let mut usage = GroupingUsage::default();
                grouping_sort(&mut values, &mut usage, |a, b, usage| {
                    usage.step();
                    a.cmp(b)
                });
                let comparisons = grouping_sort_comparisons(len).unwrap();
                assert_eq!(values, expected);
                assert!(
                    usage.comparisons <= comparisons,
                    "len={len}, pattern={pattern}"
                );
                assert!(
                    usage.work <= 3 * comparisons + len.saturating_sub(1) as u128,
                    "len={len}, pattern={pattern}"
                );
            }
        }
    }
}
