//! Synthetic THOR original-encoder clear logical-slot prefix, not checkpoint
//! authentication or an evaluated float32 profile. Two tokens, hidden width four,
//! two heads of width two: separate biased Q/K, head views, raw bidirectional QK^T.
//! Stops before scale, masks, softmax, refresh, values, residuals, or encryption.
//! Binary64 separate multiply/add in existing primitive order (no `mul_add`).
//! This clear reference is artifact-schedule-specific: projection visits hidden
//! coordinates by token-rotated within-block routes; score accumulation retains
//! wrapped/unwrapped partials before their final sum. Ascending dense contraction
//! is a geometry oracle, NOT a general bitwise binary64 contract. No HE, full-paper
//! implementation, or physical CKKS equivalence is claimed.
//! Original immutable descriptors, numeric capacities, attempt burns and ledger
//! belong to clear_input_owner. No borrowed guard, adopted Vec, caller ledger or
//! local issuer is accepted. Logical accounting is not an allocator/RSS bound.
use crate::clear_input_owner::{self as owner, BorrowedOutput, ClearInputs, Phase, Prepared};
use pllm_core::slot_matrix_layout as matrix;
use pllm_garble::shared_allocation::{ExternalCharge, MAX_ALLOCATION_BYTES};
use pllm_models::ModelOperator;
use std::{cell::Ref, mem::size_of};
use zeroize::Zeroize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Region,
    Shape,
    Policy,
    Source,
    Resource,
    Work,
    NonFinite,
    Burned,
}
fn owned(e: owner::Error) -> Error {
    match e {
        owner::Error::Resource => Error::Resource,
        owner::Error::Work => Error::Work,
        owner::Error::Shape => Error::Shape,
        owner::Error::NonFinite => Error::NonFinite,
        owner::Error::Burned => Error::Burned,
    }
}
#[derive(Clone, Copy)]
pub struct Limits {
    pub max_owner_bytes: usize,
    pub max_phase_bytes: usize,
    /// TOTAL original-owner work, including construction and previous attempts.
    pub max_work: usize,
    pub primitive: matrix::Limits,
}
pub const MAX_WORK: usize = owner::MAX_WORK;
/// Five owned tensors; indices select ONLY within the one retained owner.
#[derive(Clone, Copy)]
pub struct Bindings {
    pub input: usize,
    pub query_weights: usize,
    pub query_bias: usize,
    pub key_weights: usize,
    pub key_bias: usize,
}
impl Bindings {
    pub const fn canonical() -> Self {
        Self {
            input: 0,
            query_weights: 1,
            query_bias: 2,
            key_weights: 3,
            key_bias: 4,
        }
    }
    fn ids(self) -> [usize; 5] {
        [
            self.input,
            self.query_weights,
            self.query_bias,
            self.key_weights,
            self.key_bias,
        ]
    }
}
#[derive(Clone, Copy)]
pub enum HeadReshape {
    BatchHeadsSequenceFeature,
}
#[derive(Clone, Copy)]
pub enum ScoreOutput {
    RawBidirectionalUnscaled,
}
pub struct ClearReferencePolicy<'a> {
    original: &'a ClearInputs,
    bindings: Bindings,
}
impl<'a> ClearReferencePolicy<'a> {
    /// Explicit synthetic float64 policy; no inferred BERT/source-model factory.
    pub fn encoder_prefix(
        original: &'a ClearInputs,
        bindings: Bindings,
        _: HeadReshape,
        _: ScoreOutput,
    ) -> Self {
        Self { original, bindings }
    }
}
/// Bounded schema/dependency checks, NOT the old opaque JSON owner census.
/// The shared builder already pays and enforces capacities, unique bounded names,
/// rank<=4, inputs<=2, attrs<=8 and key<=64 BEFORE materializing them. Budget:
/// namespace comparisons1280 + dependencies1536 + tensor-name bindings1024 +
/// bounded schema/lookups2048 + views/geometry/control init and retirement2304.
pub const PREFLIGHT_CONTROL: usize = 1280 + 1536 + 1024 + 2048 + 2304;
/// Same numeric visits/arithmetic as the original primitive-order prefix:
/// geometry128; preparation80; packs48; projections336; biases48; transposes48;
/// unpacks72; replication16; corrected MM136; finite24; init/retirement288.
pub const CLEAR_WORK: usize = 128 + 80 + 48 + 336 + 48 + 48 + 72 + 16 + 136 + 24 + 288;
#[derive(Clone, Copy)]
pub struct Quote {
    pub original_bytes: usize,
    pub phase_bytes: usize,
    pub clear_work: usize,
    pub control_work: usize,
    pub projection_schedule: matrix::Schedule,
    pub score_schedule: matrix::Schedule,
}
pub struct QuotedPrefix<'a> {
    policy: ClearReferencePolicy<'a>,
    geometry: Quote,
    // Retired last, including failed validation and dropped unused quotes.
    phase: Phase<'a, Scratch>,
}
impl QuotedPrefix<'_> {
    pub fn resources(&self) -> Quote {
        self.geometry
    }
}
struct Scratch {
    input_heads: [f64; 8],
    input_slots: [f64; 8],
    weights: [[f64; 16]; 2],
    affine_slots: [[f64; 8]; 2],
    head_slots: [[f64; 8]; 2],
    heads: [[f64; 8]; 2],
    key_transposed: [f64; 8],
    replicated_key: [f64; 8],
    mm: [f64; 16],
    score_slots: [f64; 8],
    scores: [f64; 8],
}
impl Scratch {
    fn empty() -> Self {
        Self {
            input_heads: [0.; 8],
            input_slots: [0.; 8],
            weights: [[0.; 16]; 2],
            affine_slots: [[0.; 8]; 2],
            head_slots: [[0.; 8]; 2],
            heads: [[0.; 8]; 2],
            key_transposed: [0.; 8],
            replicated_key: [0.; 8],
            mm: [0.; 16],
            score_slots: [0.; 8],
            scores: [0.; 8],
        }
    }
}
impl Zeroize for Scratch {
    fn zeroize(&mut self) {
        self.input_heads.zeroize();
        self.input_slots.zeroize();
        self.weights.zeroize();
        self.affine_slots.zeroize();
        self.head_slots.zeroize();
        self.heads.zeroize();
        self.key_transposed.zeroize();
        self.replicated_key.zeroize();
        self.mm.zeroize();
        self.score_slots.zeroize();
        self.scores.zeroize();
    }
}
// Standard consuming cleanup guard: erase on failure, transfer once on success.
// No replacement Scratch is initialized at transfer, and no double retirement.
struct Workspace(Option<Scratch>);
impl Drop for Workspace {
    fn drop(&mut self) {
        if let Some(s) = &mut self.0 {
            s.zeroize();
        }
    }
}
/// Output retains the actual immutable backing, not a same-budget guard label.
/// Positive public API/type control:
/// ```
/// use pllm_compiler::clear_attention_prefix::{execute, Bindings, ClearReferencePolicy, HeadReshape, Limits, Output, ScoreOutput};
/// use pllm_compiler::clear_input_owner::ClearInputs;
/// fn run(input: &ClearInputs, caps: Limits) -> Output<'_> {
///     execute(ClearReferencePolicy::encoder_prefix(input, Bindings::canonical(),
///         HeadReshape::BatchHeadsSequenceFeature, ScoreOutput::RawBidirectionalUnscaled), caps).unwrap()
/// }
/// ```
/// No fresh-ledger injection:
/// ```compile_fail,E0061
/// use pllm_compiler::clear_attention_prefix::{execute, ClearReferencePolicy, Limits};
/// use pllm_core::row_fold::WorkLedger;
/// fn reset(p: ClearReferencePolicy<'_>, caps: Limits, fresh: &mut WorkLedger) {
///     let _ = execute(p, caps, fresh);
/// }
/// ```
/// No borrowed-guard substitution:
/// ```compile_fail,E0061
/// use pllm_compiler::clear_attention_prefix::{execute, ClearReferencePolicy, Limits};
/// use pllm_garble::shared_allocation::ExternalCharge;
/// fn replace(p: ClearReferencePolicy<'_>, caps: Limits, guard: &ExternalCharge<'_>) {
///     let _ = execute(p, caps, guard);
/// }
/// ```
/// Actual original cannot retire while output remains live:
/// ```compile_fail,E0505
/// use pllm_compiler::clear_attention_prefix::{execute, Bindings, ClearReferencePolicy, HeadReshape, Limits, ScoreOutput};
/// use pllm_compiler::clear_input_owner::ClearInputs;
/// fn retire(input: ClearInputs, caps: Limits) {
///     let out = execute(ClearReferencePolicy::encoder_prefix(&input, Bindings::canonical(),
///         HeadReshape::BatchHeadsSequenceFeature, ScoreOutput::RawBidirectionalUnscaled), caps).unwrap();
///     drop(input);
///     std::hint::black_box(out);
/// }
/// ```
/// No original payload mutation behind immutable views:
/// ```compile_fail,E0594
/// use pllm_compiler::clear_input_owner::ClearInputs;
/// fn mutate(input: &ClearInputs) {
///     let mut values = input.f64_tensor(0).unwrap();
///     values[0] = 7.0;
/// }
/// ```
/// No replacement behind a retained output:
/// ```compile_fail,E0506
/// use pllm_compiler::clear_attention_prefix::{execute, Bindings, ClearReferencePolicy, HeadReshape, Limits, ScoreOutput};
/// use pllm_compiler::clear_input_owner::ClearInputs;
/// fn replace(input: &mut ClearInputs, replacement: ClearInputs, caps: Limits) {
///     let out = execute(ClearReferencePolicy::encoder_prefix(input, Bindings::canonical(),
///         HeadReshape::BatchHeadsSequenceFeature, ScoreOutput::RawBidirectionalUnscaled), caps).unwrap();
///     *input = replacement;
///     std::hint::black_box(out);
/// }
/// ```
pub struct Output<'a> {
    inner: BorrowedOutput<'a, Scratch>,
}
impl Output<'_> {
    /// [head, query token, key token], no scale or causal restriction.
    pub fn values(&self) -> &[f64; 8] {
        &self.inner.values().scores
    }
    /// [head, token, feature], biased projection outputs.
    pub fn query(&self) -> &[f64; 8] {
        &self.inner.values().heads[0]
    }
    pub fn key(&self) -> &[f64; 8] {
        &self.inner.values().heads[1]
    }
    pub fn replicated_key_slots(&self) -> &[f64; 8] {
        &self.inner.values().replicated_key
    }
    pub fn original(&self) -> &ClearInputs {
        self.inner.original()
    }
}
fn finite(v: f64) -> Result<f64, Error> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Error::NonFinite)
    }
}
fn mat(e: matrix::Error) -> Error {
    if e == matrix::Error::NonFinite {
        Error::NonFinite
    } else {
        Error::Resource
    }
}
fn primitive() -> matrix::Limits {
    matrix::Limits {
        max_buffer_elements: 16,
        max_work: 80,
    }
}
fn layout() -> Result<matrix::Layout, Error> {
    matrix::Layout::new(2, 2, 2, 2, 2, 1, matrix::Diagonal::Lower, primitive()).map_err(mat)
}
/// Complete phase inventory before metadata views/schema/layouts/workspace.
/// Includes staging/moves, controls, five numeric Refs, metadata Refs, cleanup
/// and primitive temporaries. Shared Prepared adds its own wrapper/guard bytes.
pub const PHASE_BYTES: usize = 2 * size_of::<Output<'static>>()
    + 2 * size_of::<Scratch>()
    + 2 * size_of::<Workspace>()
    + 2 * size_of::<QuotedPrefix<'static>>()
    + 4 * size_of::<Quote>()
    + 4 * size_of::<matrix::Layout>()
    + 4 * size_of::<matrix::Schedule>()
    + 5 * size_of::<Ref<'static, [f64]>>()
    + 4 * size_of::<Ref<'static, [pllm_models::ModelOperation]>>()
    + 4 * size_of::<Ref<'static, [owner::ClearTensor]>>()
    + size_of::<(ClearReferencePolicy<'static>, Limits)>()
    + 16 * size_of::<(usize, std::ops::Range<usize>, Option<usize>)>()
    + 16 * size_of::<Result<f64, Error>>()
    + 16 * size_of::<Result<usize, Error>>()
    + 4 * size_of::<Result<QuotedPrefix<'static>, Error>>()
    + size_of::<Result<Output<'static>, Error>>()
    + 2 * size_of::<[f64; 16]>();
/// Exact additional charge made by the existing shared Prepared::reserve seam.
pub const PHASE_CHARGED_BYTES: usize = PHASE_BYTES
    + size_of::<BorrowedOutput<'static, Scratch>>()
    + size_of::<Prepared<'static>>()
    + size_of::<ExternalCharge<'static>>();

fn validate(p: &ClearReferencePolicy<'_>, cap: Limits) -> Result<Quote, Error> {
    let original_bytes = p.original.charged_bytes();
    if cap.max_owner_bytes > MAX_ALLOCATION_BYTES
        || original_bytes > cap.max_owner_bytes
        || cap.max_phase_bytes > MAX_ALLOCATION_BYTES
        || PHASE_CHARGED_BYTES > cap.max_phase_bytes
        || cap.primitive.max_buffer_elements < 16
        || cap.primitive.max_work < 80
    {
        return Err(Error::Resource);
    }
    let tensors = p.original.tensors();
    let ops = p.original.operations();
    if tensors.len() != 5 || ops.len() != 5 {
        return Err(Error::Region);
    }
    let ids = p.bindings.ids();
    for (i, &id) in ids.iter().enumerate() {
        if id >= tensors.len() || ids[..i].contains(&id) {
            return Err(Error::Source);
        }
        let shape: &[usize] = if i == 0 {
            &[1, 2, 4]
        } else if i % 2 == 1 {
            &[4, 4]
        } else {
            &[4]
        };
        if tensors[id].shape() != shape {
            return Err(Error::Shape);
        }
        let values = p.original.f64_tensor(id).map_err(owned)?;
        if values.len()
            != if i == 0 {
                8
            } else if i % 2 == 1 {
                16
            } else {
                4
            }
        {
            return Err(Error::Shape);
        }
    }
    let operators = [
        ModelOperator::Linear,
        ModelOperator::Linear,
        ModelOperator::Reshape,
        ModelOperator::Reshape,
        ModelOperator::AttentionScores,
    ];
    let input_name = tensors[p.bindings.input].name();
    for (i, op) in ops.iter().enumerate() {
        if op.operator != operators[i]
            || op.layer != ops[0].layer
            || op.state_kind.is_some()
            || op.id == input_name
        {
            return Err(Error::Region);
        }
        let shape: &[u64] = if i < 2 { &[1, 2, 4] } else { &[1, 2, 2, 2] };
        if op.output_shape != shape || op.inputs.len() != if i == 4 { 2 } else { 1 } {
            return Err(Error::Shape);
        }
        for (j, input) in op.inputs.iter().enumerate() {
            let expected = if i < 2 {
                input_name
            } else if i < 4 {
                ops[i - 2].id.as_str()
            } else {
                ops[j + 2].id.as_str()
            };
            if input != expected {
                return Err(Error::Source);
            }
        }
        let attrs = op.attributes.as_object().ok_or(Error::Policy)?;
        let expected_attributes = if i < 2 {
            7
        } else if i < 4 {
            2
        } else {
            1
        };
        // O(1) cardinality gate BEFORE any traversal or keyed lookup. Shared
        // construction independently enforces <=8 attributes and <=64-byte keys.
        if attrs.len() != expected_attributes {
            return Err(Error::Policy);
        }
        if attrs.keys().any(|k| k.len() > 64) {
            return Err(Error::Policy);
        }
        if i < 2 {
            if attrs.get("weight").and_then(|v| v.as_str()) != Some(tensors[ids[1 + i * 2]].name())
                || attrs.get("bias").and_then(|v| v.as_str())
                    != Some(tensors[ids[2 + i * 2]].name())
                || attrs.get("compute_dtype").and_then(|v| v.as_str()) != Some("float64")
                || attrs.get("weight_layout").and_then(|v| v.as_str()) != Some("input_output")
                || attrs.get("formula").and_then(|v| v.as_str())
                    != Some("x_matmul_weight_plus_bias")
            {
                return Err(Error::Policy);
            }
            for (key, expected) in [
                ("weight_shape", &[4u64, 4][..]),
                ("bias_shape", &[4u64][..]),
            ] {
                let arr = attrs
                    .get(key)
                    .and_then(|v| v.as_array())
                    .ok_or(Error::Policy)?;
                if arr.len() != expected.len()
                    || arr
                        .iter()
                        .zip(expected)
                        .any(|(a, b)| a.as_u64() != Some(*b))
                {
                    return Err(Error::Shape);
                }
            }
        } else if i < 4 {
            if attrs.get("layout").and_then(|v| v.as_str()) != Some("batch_heads_sequence_feature")
                || attrs.get("input_layout").and_then(|v| v.as_str())
                    != Some("batch_sequence_hidden")
            {
                return Err(Error::Policy);
            }
        } else if attrs.get("group_size").and_then(|v| v.as_u64()) != Some(1) {
            return Err(Error::Policy);
        }
    }
    let l = layout()?;
    Ok(Quote {
        original_bytes,
        phase_bytes: PHASE_CHARGED_BYTES,
        clear_work: CLEAR_WORK,
        control_work: PREFLIGHT_CONTROL,
        projection_schedule: matrix::block_pc_schedule(l).map_err(mat)?,
        score_schedule: matrix::optimized_lower_schedule(l).map_err(mat)?,
    })
}
/// Burn one shared-owner ticket; prepay ALL work on its original ledger BEFORE
/// views; reserve complete phase BEFORE schema validation/layouts/workspace.
/// No callback allocates buffers. Unused/dropped quotes do not refund work.
pub fn quote(policy: ClearReferencePolicy<'_>, cap: Limits) -> Result<QuotedPrefix<'_>, Error> {
    let ticket = policy.original.begin_attempt().map_err(owned)?;
    let consumed = policy.original.consumed_work();
    let (_, prepared) = ticket
        .with_ledger(|ledger| {
            if cap.max_work > MAX_WORK
                || consumed
                    .checked_add(PREFLIGHT_CONTROL + CLEAR_WORK)
                    .filter(|n| *n <= cap.max_work)
                    .is_none()
            {
                return Err(owner::Error::Work);
            }
            ledger
                .debit(CLEAR_WORK, PREFLIGHT_CONTROL)
                .map_err(|_| owner::Error::Work)
        })
        .map_err(owned)?;
    let phase = prepared.reserve::<Scratch>(PHASE_BYTES).map_err(owned)?;
    let admitted = (policy, phase);
    let geometry = validate(&admitted.0, cap)?;
    Ok(QuotedPrefix {
        policy: admitted.0,
        geometry,
        phase: admitted.1,
    })
}
pub fn execute(policy: ClearReferencePolicy<'_>, cap: Limits) -> Result<Output<'_>, Error> {
    quote(policy, cap)?.execute()
}
impl<'a> QuotedPrefix<'a> {
    pub fn execute(self) -> Result<Output<'a>, Error> {
        let admitted = self;
        let mut workspace = Workspace(Some(Scratch::empty()));
        let s = workspace.0.as_mut().ok_or(Error::Resource)?;
        let ids = admitted.policy.bindings.ids();
        let o = admitted.policy.original;
        let input = o.f64_tensor(ids[0]).map_err(owned)?;
        let weights = [
            o.f64_tensor(ids[1]).map_err(owned)?,
            o.f64_tensor(ids[3]).map_err(owned)?,
        ];
        let biases = [
            o.f64_tensor(ids[2]).map_err(owned)?,
            o.f64_tensor(ids[4]).map_err(owned)?,
        ];
        let l = layout()?;
        for h in 0..2 {
            for f in 0..2 {
                for t in 0..2 {
                    s.input_heads[(h * 2 + f) * 2 + t] = finite(input[t * 4 + h * 2 + f])?;
                }
            }
        }
        l.pack(&s.input_heads, &mut s.input_slots, primitive())
            .map_err(mat)?;
        for projection in 0..2 {
            // W is input/output. Algorithm 1 multiplies W^T * X^T.
            for r in 0..4 {
                for c in 0..4 {
                    s.weights[projection][r * 4 + c] = finite(weights[projection][c * 4 + r])?;
                }
            }
            matrix::block_pc_mm(
                l,
                &s.weights[projection],
                &s.input_slots,
                &mut s.affine_slots[projection],
                primitive(),
            )
            .map_err(mat)?;
            for lane in 0..8 {
                let (h, feature, _) = l.lane(lane).map_err(mat)?.ok_or(Error::Shape)?;
                let bias = finite(biases[projection][h * 2 + feature])?;
                s.affine_slots[projection][lane] = finite(s.affine_slots[projection][lane] + bias)?;
            }
            matrix::transpose(
                l,
                &s.affine_slots[projection],
                l,
                &mut s.head_slots[projection],
                primitive(),
            )
            .map_err(mat)?;
            l.unpack(
                &s.head_slots[projection],
                &mut s.heads[projection],
                primitive(),
            )
            .map_err(mat)?;
        }
        matrix::transpose(l, &s.head_slots[1], l, &mut s.key_transposed, primitive())
            .map_err(mat)?;
        matrix::replicate(l, &s.key_transposed, &mut s.replicated_key, primitive()).map_err(mat)?;
        // Capacity one => two full frames; D25-corrected routes unchanged.
        matrix::optimized_lower_mm(
            l,
            &s.head_slots[0],
            l,
            &s.key_transposed,
            l,
            &mut s.mm,
            &mut s.score_slots,
            primitive(),
        )
        .map_err(mat)?;
        l.unpack(&s.score_slots, &mut s.scores, primitive())
            .map_err(mat)?;
        for v in &s.scores {
            finite(*v)?;
        }
        let payload = workspace.0.take().ok_or(Error::Resource)?;
        Ok(Output {
            inner: admitted.phase.finish(payload),
        })
    }
}

#[cfg(test)]
#[path = "clear_attention_prefix_tests.rs"]
mod tests;
