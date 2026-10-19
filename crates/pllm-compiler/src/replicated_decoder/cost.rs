//! Read-only P0 diagnostics. These records are NOT execution/admission plans.
//!
//! Shape estimates intentionally survive rejected workload/numeric admission.
//! They contain no weights, shares, executable schedule, or capacity overrides.
//! `baseline` identifies the pre-bulk, 128-lane equality lookup schedule; changing
//! executor granularity must not silently change the meaning of this report.

use super::{tensor, Code, Error, Interval, Program, WorkBudget};
use serde::Serialize;
use std::collections::BTreeMap;

pub const PROFILE: &str = "pllm.puma.shape-cost.pre-bulk.v1";

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Dimensions {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocabulary: usize,
    pub tied_embeddings: bool,
}

impl Dimensions {
    /// Shape of the common.py revision, not evidence of checkpoint execution.
    pub const COMMON_QWEN: Self = Self {
        layers: 24,
        hidden: 896,
        intermediate: 4864,
        heads: 14,
        kv_heads: 2,
        head_dim: 64,
        vocabulary: 151936,
        tied_embeddings: true,
    };

    fn validate(self) -> Result<(), Error> {
        if [
            self.layers,
            self.hidden,
            self.intermediate,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.vocabulary,
        ]
        .contains(&0)
            || self.heads.checked_mul(self.head_dim) != Some(self.hidden)
            || self.heads % self.kv_heads != 0
            || self.hidden > 8192
            || self.intermediate > 8192
            || self.vocabulary > 262144
            || self.layers > 64
        {
            return Err(Error::Shape);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Work {
    pub ordinals: u64,
    pub public_prss_blocks: u64,
    pub pair_prss_blocks: u64,
    pub private_prss_blocks_upper: u64,
    pub aes_blocks_per_party_upper: u64,
}

impl From<WorkBudget> for Work {
    fn from(b: WorkBudget) -> Self {
        let c = b.cursors();
        Self {
            ordinals: b.operations(),
            public_prss_blocks: c.public_blocks,
            pair_prss_blocks: c.pair_blocks,
            private_prss_blocks_upper: c.private_blocks,
            aes_blocks_per_party_upper: c.aes_work,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Category {
    pub name: &'static str,
    pub semantic_invocations: u64,
    pub output_lanes: u64,
    /// Native WorkBudget of included primitives only. AES is that subset's
    /// per-party upper bound, NOT a lower bound on whole-program AES work.
    pub included_work: Work,
    pub omitted_work: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct LinkFloor {
    /// Payload only on EACH of the three directed rotation-cycle edges.
    /// The other three directions are not assigned this bound.
    pub bytes_per_rotation_edge: u64,
    pub frames_per_rotation_edge: u64,
    /// Critical sequential waves of lookup A2B alone, not ordinals or HTTP calls.
    pub lookup_a2b_waves: u64,
    pub seconds_at_100_mbps: f64,
    pub seconds_at_40_mbps: f64,
    pub exceeds_1200_seconds_at_40_mbps: bool,
}

#[derive(Debug, Serialize)]
pub struct ShapeReport {
    pub profile: &'static str,
    pub executable: bool,
    pub dimensions: Dimensions,
    pub input_rows: usize,
    pub selections: usize,
    pub prefix_rows: Vec<usize>,
    pub recomputed_rows: u64,
    pub categories: Vec<Category>,
    pub included_work: Work,
    /// Public matrix/lookup products only; excludes norm/rotary local affines.
    pub public_ring64_macs_per_worker: u64,
    pub body_public_ring64_macs_per_worker: u64,
    pub head_public_ring64_macs_per_worker: u64,
    pub lookup_public_ring64_macs_per_worker: u64,
    pub secret_attention_dot_products: u64,
    pub logical_public_matrix_read_bytes_per_worker: u64,
    /// Optimistic single lookup table scan per selection, not per private row.
    pub bulk_logical_public_matrix_read_bytes_per_worker: u64,
    pub public_matrix_snapshot_bytes_per_worker: u64,
    pub largest_secret_owner_values: u64,
    pub largest_secret_owner_bytes: u64,
    pub minimum_program_node_count: u64,
    pub minimum_node_header_bytes: u64,
    pub peer_payload_floor: LinkFloor,
    pub unknowns: Vec<&'static str>,
}

fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b).ok_or(Error::Bounds)
}

/// Apply actual 128-lane primitive tiling without allocating output tensors.
fn tiled(n: u64, f: impl Fn(usize) -> Result<WorkBudget, Error>) -> Result<WorkBudget, Error> {
    let mut result = WorkBudget::default();
    if n >= 128 {
        result = f(128)?.checked_mul(n / 128)?;
    }
    if n % 128 != 0 {
        result = result.checked_add(f((n % 128) as usize)?)?;
    }
    Ok(result)
}

fn rescale(n: usize) -> Result<WorkBudget, Error> {
    WorkBudget::rescale(n)
}

/// Full D9 public-weight shape inventory. Never calls compiler admission,
/// imports checkpoints, or returns an executable Program. Limits here bound
/// diagnostic arithmetic/allocation only, not serving capacity.
pub fn estimate(d: Dimensions, rows: usize, selections: usize) -> Result<ShapeReport, Error> {
    d.validate()?;
    if rows == 0 || rows > 64 || selections == 0 || selections > 8 {
        return Err(Error::Shape);
    }
    let prefix_rows: Vec<_> = (0..selections).map(|s| rows + s).collect();
    let r = prefix_rows.iter().try_fold(0, |a, &b| add(a, b as u64))?;
    let (l, h, i, v, s) = (
        d.layers as u64,
        d.hidden as u64,
        d.intermediate as u64,
        d.vocabulary as u64,
        selections as u64,
    );
    let kv = mul(d.kv_heads as u64, d.head_dim as u64)?;
    let body_weights = mul(
        l,
        add(
            add(mul(2, mul(h, h)?)?, mul(2, mul(h, kv)?)?)?,
            mul(3, mul(h, i)?)?,
        )?,
    )?;
    let table = mul(v, h)?;
    let lookup_macs = mul(2, mul(r, table)?)?;
    let body_macs = mul(2, mul(r, body_weights)?)?;
    let head_macs = mul(2, mul(s, table)?)?;
    let scans = add(mul(r, table)?, mul(s, add(body_weights, table)?)?)?;
    let bulk_scans = mul(s, add(body_weights, mul(2, table)?)?)?;
    let snapshot = mul(
        8,
        add(
            body_weights,
            mul(if d.tied_embeddings { 1 } else { 2 }, table)?,
        )?,
    )?;
    let lookup_bounds = Interval::new(-(v as i64 - 1), v as i64 - 1)?;
    let lookup = tiled(v, |n| {
        Ok(tensor::Program::equal_zero(n, lookup_bounds)?.work_budget())
    })?
    .checked_mul(r)?;
    let mut categories = Vec::new();
    let mut total = WorkBudget::default();
    let mut push = |name, invocations, lanes, work: WorkBudget, omitted| -> Result<(), Error> {
        total = total.checked_add(work)?;
        categories.push(Category {
            name,
            semantic_invocations: invocations,
            output_lanes: lanes,
            included_work: work.into(),
            omitted_work: omitted,
        });
        Ok(())
    };
    push("lookup", r, mul(r, h)?, lookup, None)?;
    let columns = add(add(mul(2, h)?, mul(2, kv)?)?, mul(2, i)?)?;
    let linear_lanes = mul(mul(l, r)?, columns)?;
    // Each matrix output has separate tail rounding, even if combined lane
    // totals happen to be divisible by 128 on common Qwen.
    let mut body = WorkBudget::default();
    for width in [h, kv, kv, h, i, i, h] {
        body = body.checked_add(tiled(width * r, rescale)?.checked_mul(l)?)?;
    }
    // columns includes gate/up but down also emits h.
    push(
        "body_public_linear",
        mul(mul(7, l)?, s)?,
        add(linear_lanes, mul(mul(l, r)?, h)?)?,
        body,
        None,
    )?;
    let norms = mul(add(mul(2, l)?, 1)?, r)?;
    let norm_floor = WorkBudget::product(d.hidden, true)?
        .checked_mul(2)?
        .checked_add(rescale(d.hidden)?)?
        .checked_add(rescale(1)?)?
        .checked_mul(norms)?;
    push(
        "rmsnorm",
        norms,
        mul(norms, h)?,
        norm_floor,
        Some("rsqrt polynomial, comparisons, local maps and checked-range guards"),
    )?;
    let mut attention = WorkBudget::default();
    let mut softmax_rows = 0;
    let mut softmax_lanes = 0;
    let mut attention_macs = 0;
    let mut rotary = WorkBudget::default();
    for &prefix in &prefix_rows {
        let n = prefix as u64;
        let factor = mul(l, d.heads as u64)?;
        for valid in 1..=prefix {
            // Current matrix tiling bounds both operands by MAX_LANES.
            let tile = (super::puma::MAX_LANES / d.head_dim).min(128);
            for start in (0..valid).step_by(tile) {
                attention = attention.checked_add(
                    WorkBudget::product(tile.min(valid - start), true)?.checked_mul(factor)?,
                )?;
            }
            let tile = (super::puma::MAX_LANES / valid).min(128);
            for start in (0..d.head_dim).step_by(tile) {
                attention = attention.checked_add(
                    WorkBudget::product(tile.min(d.head_dim - start), true)?.checked_mul(factor)?,
                )?;
            }
            softmax_lanes = add(softmax_lanes, mul(valid as u64, factor)?)?;
            attention_macs = add(
                attention_macs,
                mul(mul(2, valid as u64)?, mul(d.head_dim as u64, factor)?)?,
            )?;
        }
        softmax_rows = add(softmax_rows, mul(n, factor)?)?;
        // One rescale per coordinate is a strict subset of rotary arithmetic.
        rotary = rotary.checked_add(tiled(mul(n, add(h, kv)?)?, rescale)?.checked_mul(l)?)?;
    }
    push(
        "attention_scores_and_values",
        mul(2, softmax_rows)?,
        add(softmax_lanes, mul(softmax_rows, d.head_dim as u64)?)?,
        attention,
        None,
    )?;
    push(
        "rotary",
        mul(mul(2, l)?, s)?,
        mul(mul(l, r)?, add(h, kv)?)?,
        rotary,
        Some("remaining rotary local operations and any additional rescale"),
    )?;
    push(
        "softmax",
        softmax_rows,
        softmax_lanes,
        WorkBudget::default(),
        Some("entire softmax, score scale and checked-range guards; range-dependent"),
    )?;
    let activation_lanes = mul(mul(l, r)?, i)?;
    push(
        "silu",
        mul(l, s)?,
        activation_lanes,
        WorkBudget::default(),
        Some("entire guarded SiLU; final checked profile not yet admitted"),
    )?;
    push(
        "gate_up_multiply",
        mul(l, s)?,
        activation_lanes,
        tiled(activation_lanes, |n| WorkBudget::product(n, true))?,
        None,
    )?;
    push(
        "head",
        s,
        mul(s, v)?,
        tiled(v, rescale)?.checked_mul(s)?,
        None,
    )?;
    let mut width = d.vocabulary;
    let mut argmax = WorkBudget::default();
    let mut argmax_batches = 0;
    while width > 1 {
        for start in (0..width).step_by(256) {
            let pairs = (width - start).min(256) / 2;
            if pairs != 0 {
                let negative =
                    tensor::Program::negative(pairs, Interval::new(-1, 1)?)?.work_budget();
                argmax = argmax.checked_add(
                    negative.checked_add(WorkBudget::product(pairs, false)?.checked_mul(2)?)?,
                )?;
                argmax_batches += 1;
            }
        }
        width = width.div_ceil(2);
    }
    push(
        "stable_argmax",
        s,
        mul(s, v - 1)?,
        argmax.checked_mul(s)?,
        Some("local gathers/affines have no WorkBudget ordinals; guards not included"),
    )?;
    let a2b_tiles = mul(r, v.div_ceil(128))?;
    // Bits conversion: split n*8 + first AND n*8 + six 2n*8 rotations.
    // All eight phases send once along EACH fixed directed rotation edge.
    let bytes = mul(mul(r, v)?, 112)?;
    let largest = mul(*prefix_rows.last().unwrap() as u64, i)?.max(v);
    // Argmax alone: 2 value gathers + difference + negative tensor +
    // 3 value-select nodes + 2 ID gathers + 3 ID-select nodes = 12.
    // Excludes the public indices owner and every other decoder node.
    let minimum_nodes = mul(mul(argmax_batches, s)?, 12)?;
    Ok(ShapeReport { profile: PROFILE, executable: false, dimensions: d,
        input_rows: rows, selections, prefix_rows, recomputed_rows: r, categories,
        included_work: total.into(), public_ring64_macs_per_worker: add(add(lookup_macs, body_macs)?, head_macs)?,
        body_public_ring64_macs_per_worker: body_macs, head_public_ring64_macs_per_worker: head_macs,
        lookup_public_ring64_macs_per_worker: lookup_macs, secret_attention_dot_products: attention_macs,
        logical_public_matrix_read_bytes_per_worker: mul(scans, 8)?,
        bulk_logical_public_matrix_read_bytes_per_worker: mul(bulk_scans, 8)?,
        public_matrix_snapshot_bytes_per_worker: snapshot,
        largest_secret_owner_values: largest, largest_secret_owner_bytes: mul(largest, 16)?,
        minimum_program_node_count: minimum_nodes,
        // Archived AArch64 baseline layout, not the evolving compound Code enum.
        minimum_node_header_bytes: mul(minimum_nodes, 176)?,
        peer_payload_floor: LinkFloor { bytes_per_rotation_edge: bytes,
            frames_per_rotation_edge: mul(a2b_tiles, 8)?, lookup_a2b_waves: mul(a2b_tiles, 8)?,
            seconds_at_100_mbps: bytes as f64 * 8.0 / 100_000_000.0,
            seconds_at_40_mbps: bytes as f64 * 8.0 / 40_000_000.0,
            exceeds_1200_seconds_at_40_mbps: bytes > 6_000_000_000 },
        unknowns: vec!["whole-program work/AES until all omitted nonlinear/guard schedules admit",
            "complete peer frames, bytes, dependent waves, HTTP controls, TLS/TCP wire overhead",
            "peak tensor scratch, metadata/interning/views, simultaneous role RSS and source import overlap",
            "public bias/norm source storage and import reads; physical disk/cache traffic",
            "CPU throughput, AES throughput, HTTP latency and end-to-end elapsed time"],
    })
}

/// Separate identity: current expanded body, compact selection, eight-row lookup
/// and a hypothetical guard at EVERY RMSNorm input. Not an admission profile.
pub const CAPACITY_AUDIT_PROFILE: &str = "pllm.puma.capacity-audit.guard256.v1";

#[derive(Debug, Serialize)]
pub struct NodeCharge {
    pub category: &'static str,
    pub nodes: u64,
    /// Current Builder::push's cumulative 16*lanes policy, NOT peak live memory.
    pub cumulative_secret_value_bytes: u64,
    pub conditional: bool,
}

#[derive(Debug, Serialize)]
pub struct TemplateCharge {
    pub category: &'static str,
    pub work: Option<Work>,
    pub failure: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CapacityAudit {
    pub profile: &'static str,
    pub executable: bool,
    pub input_rows: usize,
    pub selections: usize,
    pub recomputed_rows: u64,
    pub guard_cap_real_units: u64,
    pub maximum_guard_tiles: u64,
    pub nodes: Vec<NodeCharge>,
    pub priced_unconditional_nodes: u64,
    pub priced_nodes_with_all_guards: u64,
    pub cumulative_value_bytes_without_guards: u64,
    pub cumulative_value_bytes_with_all_guards: u64,
    pub template_charges: Vec<TemplateCharge>,
    /// Conditional on successful domain proofs for the fixed schedules. Not a
    /// proof that real source coefficients fit cap256 or admit every primitive.
    pub conditional_work_with_all_guards: Option<Work>,
    pub shape_work_without_guards: Option<Work>,
    pub actual_bulk_lookup_scans: u64,
    pub logical_matrix_read_bytes_per_worker: u64,
    /// Frozen intermediate measurement: not current/final compact codec bytes.
    pub intermediate_lookup_bytes_all_directions: u64,
    pub intermediate_lookup_slowest_link_seconds_at_40_mbps_floor: f64,
    /// New compact18 fixture extrapolation. Native encoded frames only; not
    /// MessagePack/HTTP/WebSocket/TCP bytes or a complete-model measurement.
    pub compact18_lookup_encoded_bytes_all_directions: u64,
    pub compact18_lookup_encoded_frames_all_directions: u64,
    pub compact18_a2b_encoded_bytes_per_rotation_edge: u64,
    pub compact18_a2b_seconds_per_edge_at_100_mbps: f64,
    pub compact18_a2b_seconds_per_edge_at_40_mbps: f64,
    pub lookup_coordinator_advances_floor: u64,
    pub exceeds_original_coordinator_barrier_budget: bool,
    pub lookup_remote_control_calls_floor: u64,
    /// Guarded SiLU's three Sign instructions still use legacy PUM1 words.
    /// Split n + initial AND n + six doubled gates = 14*n ring64 words
    /// per sign on each rotation edge. Headers/OT/products are additional.
    pub silu_sign_payload_bytes_per_rotation_edge_floor: u64,
    pub known_stage_bytes_per_rotation_edge_floor: u64,
    pub known_stage_seconds_per_edge_at_100_mbps_floor: f64,
    pub known_stage_seconds_per_edge_at_40_mbps_floor: f64,
    pub source_numeric_proof_complete: bool,
    pub unpriced: Vec<&'static str>,
}

/// Dimension-only policy audit, bounded to common-Qwen geometry. Templates use
/// synthetic public coefficients ONLY to count their fixed native schedules;
/// no SourceTensor, model publication, executable decoder or capacity override
/// is constructed. Exact incoming ranges, source gamma and guard elisions remain
/// unknown. This is not a worst-case proof of whole-source numeric admission.
pub fn audit_common_capacity(rows: usize, selections: usize) -> Result<CapacityAudit, Error> {
    if rows == 0 || rows > 64 || selections == 0 || selections > 8 {
        return Err(Error::Shape);
    }
    let d = Dimensions::COMMON_QWEN;
    let (l, h, i, heads, hd, kv) = (24_u64, 896_u64, 4864_u64, 14_u64, 64_u64, 128_u64);
    let s = selections as u64;
    let prefixes: Vec<_> = (rows..rows + selections).collect();
    let r: u64 = prefixes.iter().map(|&n| n as u64).sum();
    let guards = 49 * r * h / 128;
    let mut nodes = Vec::new();
    let mut charge = |category, count, lanes, conditional| {
        nodes.push(NodeCharge {
            category,
            nodes: count,
            cumulative_secret_value_bytes: lanes * 16,
            conditional,
        });
    };
    let scans: u64 = prefixes.iter().map(|n| n.div_ceil(8) as u64).sum();
    charge("bulk_lookup", 2 * scans, r * (h + 1), false);
    let mut linear_lanes = 0;
    for (inner, columns) in [(h, h), (h, kv), (h, kv), (h, h), (h, i), (h, i), (i, h)] {
        linear_lanes += l * r * (inner + 2 * columns);
    }
    charge("body_public_linear", 21 * l * s, linear_lanes, false);
    charge(
        "qkv_bias_binary",
        3 * l * r * (h + 2 * kv) / 128,
        3 * l * r * (h + 2 * kv),
        false,
    );
    charge("rmsnorm_rows", 2 * 49 * r, 2 * 49 * r * h, false);
    charge("rotary_rows", 2 * l * r * 16, 2 * l * r * (h + kv), false);
    charge("silu_loops", 2 * l * s, 2 * l * r * i, false);
    charge("gate_up_binary", 3 * l * r * i / 128, 3 * l * r * i, false);
    charge("residual_binary", 6 * l * r * h / 128, 6 * l * r * h, false);
    let (mut score_nodes, mut value_nodes, mut score_lanes, mut value_lanes) = (0, 0, 0, 0);
    let (mut scale_nodes, mut scale_lanes, mut softmax_lanes) = (0, 0, 0);
    let factor = l * heads;
    for &n in &prefixes {
        for valid in 1..=n as u64 {
            score_nodes += factor * (1 + 2 * valid.div_ceil(16));
            score_lanes += factor * (hd + valid * hd + valid);
            value_nodes += factor * (1 + 2 * hd.div_ceil((1024 / valid).min(128)));
            value_lanes += factor * (valid + valid * hd + hd);
            softmax_lanes += 2 * factor * valid;
        }
        scale_nodes += 2 * l * (heads * n as u64 * n as u64).div_ceil(128);
        scale_lanes += 2 * l * heads * n as u64 * n as u64;
    }
    charge("attention_score_rows", score_nodes, score_lanes, false);
    charge("attention_value_rows", value_nodes, value_lanes, false);
    charge("attention_scale_tiles", scale_nodes, scale_lanes, false);
    charge("softmax_rows", 2 * factor * r, softmax_lanes, false);
    charge("head", 3 * s, s * (h + 2 * d.vocabulary as u64), false);
    charge(
        "tiled_selection",
        2 * s,
        s * (d.vocabulary as u64 + 1),
        false,
    );
    // gather(n), checked(2n), safe(n), valid(n), all_valid(1), sticky(1).
    charge(
        "all_rms_input_guards",
        6 * guards - 1,
        5 * guards * 128 + 2 * guards - 1,
        true,
    );
    let unconditional_nodes = nodes
        .iter()
        .filter(|n| !n.conditional)
        .map(|n| n.nodes)
        .sum();
    let all_nodes = nodes.iter().map(|n| n.nodes).sum();
    let value_bytes = nodes
        .iter()
        .filter(|n| !n.conditional)
        .map(|n| n.cumulative_secret_value_bytes)
        .sum();
    let all_value_bytes = nodes.iter().map(|n| n.cumulative_secret_value_bytes).sum();

    let bound = 256_i64 << 18;
    let interval = Interval::new(-bound, bound)?;
    let mut template_charges = Vec::new();
    let mut total = Some(WorkBudget::default());
    let mut record = |category, work: Result<WorkBudget, Error>| match work {
        Ok(work) => {
            total = total.and_then(|n| n.checked_add(work).ok());
            template_charges.push(TemplateCharge {
                category,
                work: Some(work.into()),
                failure: None,
            });
        }
        Err(error) => {
            total = None;
            template_charges.push(TemplateCharge {
                category,
                work: None,
                failure: Some(format!("{error:?}")),
            });
        }
    };
    let lookup_bounds = Interval::new(-151935, 151935)?;
    record(
        "bounded18_lookup",
        tiled(151936, |n| {
            Ok(tensor::Program::equal_zero_bounded(n, lookup_bounds)?.work_budget())
        })?
        .checked_mul(r),
    );
    let mut linear = WorkBudget::default();
    for columns in [h, kv, kv, h, i, i, h] {
        linear = linear.checked_add(tiled(r * columns, rescale)?.checked_mul(l)?)?;
    }
    record("body_linear_rescales", Ok(linear));
    record(
        "rmsnorm_cap256_unit_gamma_template",
        tensor::Program::rms_norm_certified(1, 896, interval, &vec![1 << 18; 896])
            .and_then(|(p, _)| p.work_budget().checked_mul(49 * r)),
    );
    record(
        "guarded_silu_template",
        tiled(l * r * i, |n| {
            Ok(
                tensor::Program::activation_guarded(tensor::Activation::Silu, n, interval)?
                    .work_budget(),
            )
        }),
    );
    record(
        "gate_up_products",
        tiled(l * r * i, |n| WorkBudget::product(n, true)),
    );
    let mut attention = WorkBudget::default();
    let mut softmax = Ok(WorkBudget::default());
    let mut softmax_templates = Vec::new();
    for valid in 1..=*prefixes.last().unwrap() {
        softmax_templates.push(
            tensor::Program::softmax_certified(1, valid, interval).map(|(p, _)| p.work_budget()),
        );
    }
    let mut scale = WorkBudget::default();
    for &n in &prefixes {
        for valid in 1..=n {
            for start in (0..valid).step_by(16) {
                attention = attention.checked_add(
                    WorkBudget::product((valid - start).min(16), true)?.checked_mul(factor)?,
                )?;
            }
            let tile = (1024 / valid).min(128);
            for start in (0..64).step_by(tile) {
                attention = attention.checked_add(
                    WorkBudget::product((64 - start).min(tile), true)?.checked_mul(factor)?,
                )?;
            }
            softmax = softmax
                .and_then(|w| w.checked_add(softmax_templates[valid - 1]?.checked_mul(factor)?));
        }
        // One local affine + one rescale per score tile, including masked slots.
        let scores = heads * n as u64 * n as u64;
        scale = scale.checked_add(
            tiled(scores, |lanes| {
                let mut b = tensor::Builder::new(lanes, Interval::new(0, 0)?)?;
                let a = b.affine(
                    b.input(),
                    None,
                    super::Affine {
                        left: 1,
                        right: 0,
                        constant: 0,
                    },
                )?;
                let out = b.rescale(a, 18)?;
                Ok(b.finish(out)?.work_budget())
            })?
            .checked_mul(l)?,
        )?;
    }
    record("causal_attention_products", Ok(attention));
    record("attention_scale", Ok(scale));
    record("softmax_cap256_template", softmax);
    record(
        "rotary_fixed_schedule",
        tensor::Program::rotary_half(1, 64, interval, &[1 << 18; 32], &[1 << 18; 32])
            .and_then(|p| p.work_budget().checked_mul(l * r * 16)),
    );
    record("full_head_rescales", tiled(151936, rescale)?.checked_mul(s));
    record(
        "tiled_stable_selection",
        super::runtime::selection::Plan::new(151936, interval)
            .and_then(|p| p.work_budget().checked_mul(s)),
    );
    let without_guards = total;
    let guard =
        tensor::Program::checked_range(128, Interval::new(i64::MIN, i64::MAX)?, interval, 0)?;
    let guard_work = guard
        .work_budget()
        .checked_add(tensor::Program::all_valid(128)?.work_budget())?
        .checked_mul(guards)?
        .checked_add(WorkBudget::product(1, false)?.checked_mul(guards - 1)?)?;
    total = total.and_then(|w| w.checked_add(guard_work).ok());
    template_charges.push(TemplateCharge {
        category: "all_rms_input_guards",
        work: Some(guard_work.into()),
        failure: None,
    });
    let body_weights = l * (2 * h * h + 2 * h * kv + 3 * h * i);
    let table = 151936 * h;
    // 60,000 bytes was measured across ALL six directions for one old-codec
    // bounded18 tile. Six-link pigeonhole floor only, not each cycle edge.
    let intermediate = r * 1187 * 60000;
    let predicate = tensor::Program::equal_zero_bounded(128, lookup_bounds)?;
    let tiles = r * 1187;
    let lookup_steps = tiles * predicate.instructions() as u64;
    let compact_a2b = tiles * 4100;
    let silu_sign_bytes = l * r * i * 3 * 112;
    let combined_bytes = silu_sign_bytes + compact_a2b;
    Ok(CapacityAudit {
        profile: CAPACITY_AUDIT_PROFILE,
        executable: false,
        input_rows: rows,
        selections,
        recomputed_rows: r,
        guard_cap_real_units: 256,
        maximum_guard_tiles: guards,
        nodes,
        priced_unconditional_nodes: unconditional_nodes,
        priced_nodes_with_all_guards: all_nodes,
        cumulative_value_bytes_without_guards: value_bytes,
        cumulative_value_bytes_with_all_guards: all_value_bytes,
        template_charges,
        conditional_work_with_all_guards: total.map(Into::into),
        shape_work_without_guards: without_guards.map(Into::into),
        actual_bulk_lookup_scans: scans,
        logical_matrix_read_bytes_per_worker: 8 * (scans * table + s * (body_weights + table)),
        intermediate_lookup_bytes_all_directions: intermediate,
        intermediate_lookup_slowest_link_seconds_at_40_mbps_floor: intermediate as f64 / 6.0 * 8.0
            / 40_000_000.0,
        compact18_lookup_encoded_bytes_all_directions: tiles * 19920,
        compact18_lookup_encoded_frames_all_directions: tiles * 40,
        compact18_a2b_encoded_bytes_per_rotation_edge: compact_a2b,
        compact18_a2b_seconds_per_edge_at_100_mbps: compact_a2b as f64 * 8.0 / 100_000_000.0,
        compact18_a2b_seconds_per_edge_at_40_mbps: compact_a2b as f64 * 8.0 / 40_000_000.0,
        lookup_coordinator_advances_floor: lookup_steps + 1,
        exceeds_original_coordinator_barrier_budget: lookup_steps + 1 > 1 << 20,
        // Four remote controls per noncomplete advance (advance + terminal
        // prepare), four per A2B wave (prepare + dispatch), two final advances.
        // Other equality waves, other stages, seal/setup/finish are additional.
        lookup_remote_control_calls_floor: 4 * lookup_steps + 4 * 7 * tiles + 2,
        silu_sign_payload_bytes_per_rotation_edge_floor: silu_sign_bytes,
        known_stage_bytes_per_rotation_edge_floor: combined_bytes,
        known_stage_seconds_per_edge_at_100_mbps_floor: combined_bytes as f64 * 8.0 / 100_000_000.0,
        known_stage_seconds_per_edge_at_40_mbps_floor: combined_bytes as f64 * 8.0 / 40_000_000.0,
        source_numeric_proof_complete: false,
        unpriced: vec![
            "source weights/gamma, exact incoming intervals and actual guard elisions",
            "source owners, prefix plumbing, compiler views/certificates/interning metadata",
            "complete-model compact bytes/frames/waves, transport envelopes/controls and CPU/RSS",
            "an admitted complete graph and a cryptographic PRSS namespace review",
        ],
    })
}

/// Exact public counters from an ALREADY admitted native program. No method on
/// this report can turn rejected source metadata into an execution plan.
#[derive(Debug, Serialize)]
pub struct AdmittedReport {
    pub work: Work,
    pub nodes: usize,
    pub program_payload_bytes: usize,
    pub tensor_invocations: u64,
    pub tensor_instructions: u64,
    /// Complete for the recognized fixed operators; new Code variants must be
    /// explicitly priced before this module compiles.
    pub tensor_instruction_count_complete: bool,
    pub compound_predicate_invocations: u64,
    pub nodes_by_code: BTreeMap<&'static str, u64>,
    pub work_by_code: BTreeMap<&'static str, Work>,
    pub compound_plan_metadata_bytes: u64,
    pub lookup_unique_program_metadata_bytes: u64,
    pub largest_compound_workspace_bytes: u64,
    /// Native start steps plus one completion advance; not peer gate rounds.
    pub scheduled_coordinator_advances: u64,
    pub stable_selection_comparisons: u64,
    pub stable_selection_products: u64,
    pub largest_tensor_admitted_bytes: usize,
    pub largest_secret_owner_values: u64,
    /// Outer retained owners only, before retiring consumed inputs. Tensor
    /// scratch, operand copies, queues and exported frames remain additional.
    pub peak_retained_secret_bytes: u64,
    pub unpriced_native_nodes: u64,
    pub public_ring64_macs_per_worker: u64,
    pub logical_public_file_read_bytes_per_worker: u64,
    pub logical_public_scalar_gather_read_bytes_per_worker: u64,
    pub peer_frames: Option<u64>,
    pub peer_rounds: Option<u64>,
    pub peer_bytes: Option<u64>,
}

pub fn admitted(program: &Program) -> Result<AdmittedReport, Error> {
    let mut report = AdmittedReport {
        work: program.work_budget().into(),
        nodes: program.nodes.len(),
        program_payload_bytes: program.payload_bytes(),
        tensor_invocations: 0,
        tensor_instructions: 0,
        tensor_instruction_count_complete: true,
        compound_predicate_invocations: 0,
        nodes_by_code: BTreeMap::new(),
        work_by_code: BTreeMap::new(),
        compound_plan_metadata_bytes: 0,
        lookup_unique_program_metadata_bytes: 0,
        largest_compound_workspace_bytes: 0,
        scheduled_coordinator_advances: 1,
        stable_selection_comparisons: 0,
        stable_selection_products: 0,
        largest_tensor_admitted_bytes: 0,
        largest_secret_owner_values: 0,
        peak_retained_secret_bytes: 0,
        unpriced_native_nodes: 0,
        public_ring64_macs_per_worker: 0,
        logical_public_file_read_bytes_per_worker: 0,
        logical_public_scalar_gather_read_bytes_per_worker: 0,
        peer_frames: None,
        peer_rounds: None,
        peer_bytes: None,
    };
    fn tensor(report: &mut AdmittedReport, p: &tensor::Program, count: u64) -> Result<(), Error> {
        if count == 0 {
            return Ok(());
        }
        report.tensor_invocations = add(report.tensor_invocations, count)?;
        report.tensor_instructions = add(
            report.tensor_instructions,
            mul(p.instructions() as u64, count)?,
        )?;
        report.largest_tensor_admitted_bytes =
            report.largest_tensor_admitted_bytes.max(p.admitted_bytes());
        report.scheduled_coordinator_advances = add(
            report.scheduled_coordinator_advances,
            mul(p.instructions() as u64, count)?,
        )?;
        Ok(())
    }
    let mut work_by_code = BTreeMap::<&'static str, WorkBudget>::new();
    let mut lookup_programs = std::collections::BTreeSet::new();
    for n in &program.nodes {
        let code = match &n.code {
            Code::Input(_) => "input",
            Code::PublicInput { .. } => "public_input",
            Code::PublicValues(_) => "public_values",
            Code::Gather(_) => "gather",
            Code::SourceGather(_) => "source_gather",
            Code::Affine { .. } => "affine",
            Code::Product { .. } => "product",
            Code::PublicProduct { .. } => "public_product",
            Code::Tensor { .. } => "tensor",
            Code::TensorLoop { .. } => "tensor_loop",
            Code::CheckedRange { .. } => "checked_range",
            Code::PublicLookup { .. } => "public_lookup",
            Code::Selection { .. } => "selection",
        };
        let count = report.nodes_by_code.entry(code).or_default();
        *count = add(*count, 1)?;
        let work = work_by_code.entry(code).or_default();
        *work = work.checked_add(n.work_budget()?)?;
        let public_owner =
            |owner: usize| matches!(program.nodes[owner].code, Code::PublicInput { .. });
        let scalar_words = match &n.code {
            Code::Gather(words) if words.owners().all(public_owner) => words.len() as u64,
            Code::Gather(words) if words.owners().any(public_owner) => words
                .iter()
                .filter(|&(owner, _)| public_owner(owner))
                .count()
                as u64,
            Code::SourceGather(view) if public_owner(view.owner) => n.lanes as u64,
            _ => 0,
        };
        let scalar_bytes = mul(scalar_words, 8)?;
        report.logical_public_scalar_gather_read_bytes_per_worker = add(
            report.logical_public_scalar_gather_read_bytes_per_worker,
            scalar_bytes,
        )?;
        report.logical_public_file_read_bytes_per_worker = add(
            report.logical_public_file_read_bytes_per_worker,
            scalar_bytes,
        )?;
        match &n.code {
            Code::Tensor { program, .. } | Code::CheckedRange { program, .. } => {
                tensor(&mut report, program, 1)?
            }
            Code::TensorLoop { program, tail, .. } => {
                tensor(&mut report, program, (n.lanes / program.lanes()) as u64)?;
                if let Some(tail) = tail {
                    tensor(&mut report, tail, 1)?;
                }
            }
            Code::PublicProduct {
                rows,
                inner,
                columns,
                ..
            } => {
                let weights = mul(*inner as u64, *columns as u64)?;
                report.public_ring64_macs_per_worker = add(
                    report.public_ring64_macs_per_worker,
                    mul(2, mul(*rows as u64, weights)?)?,
                )?;
                report.logical_public_file_read_bytes_per_worker = add(
                    report.logical_public_file_read_bytes_per_worker,
                    mul(weights, 8)?,
                )?;
            }
            Code::PublicLookup { plan, .. } => {
                let (rows, vocabulary, width) = plan.geometry();
                let (reads, macs) = plan.logical_cost()?;
                report.public_ring64_macs_per_worker =
                    add(report.public_ring64_macs_per_worker, macs)?;
                report.logical_public_file_read_bytes_per_worker =
                    add(report.logical_public_file_read_bytes_per_worker, reads)?;
                report.compound_predicate_invocations = add(
                    report.compound_predicate_invocations,
                    mul(
                        rows as u64,
                        vocabulary.div_ceil(super::runtime::lookup::VOCAB_TILE) as u64,
                    )?,
                )?;
                for p in plan.programs() {
                    let repeats = if p.lanes() == vocabulary.min(super::runtime::lookup::VOCAB_TILE)
                    {
                        rows * (vocabulary / super::runtime::lookup::VOCAB_TILE).max(1)
                    } else {
                        rows
                    };
                    tensor(&mut report, p, repeats as u64)?;
                    if lookup_programs.insert(std::sync::Arc::as_ptr(p) as usize) {
                        report.lookup_unique_program_metadata_bytes = add(
                            report.lookup_unique_program_metadata_bytes,
                            super::runtime::lookup::Plan::program_metadata_bytes(p) as u64,
                        )?;
                    }
                }
                let metadata = plan.shallow_metadata_bytes();
                report.compound_plan_metadata_bytes =
                    add(report.compound_plan_metadata_bytes, metadata as u64)?;
                let workspace = std::mem::size_of::<super::runtime::lookup::BulkLookup>()
                    + plan.predicate_workspace_bytes()
                    + 16 * rows * width
                    + 16 * rows * super::runtime::lookup::VOCAB_TILE
                    + 32 * super::runtime::lookup::VOCAB_TILE
                    + 65536
                    + 4096;
                report.largest_compound_workspace_bytes = report
                    .largest_compound_workspace_bytes
                    .max(workspace as u64);
            }
            Code::Selection { plan, .. } => {
                let (comparisons, products, tiles) = plan.logical_cost();
                report.stable_selection_comparisons =
                    add(report.stable_selection_comparisons, comparisons)?;
                report.stable_selection_products = add(report.stable_selection_products, products)?;
                report.compound_predicate_invocations =
                    add(report.compound_predicate_invocations, tiles)?;
                // Selection::Plan constructs negative(pairs, difference) for
                // each tile: its instruction count is independent of width or
                // input interval (Sign, FromBit; input is not an instruction).
                // Workspace comes from
                // the actual plan, not this instruction-shape witness.
                let predicate = tensor::Program::negative(1, Interval::new(-1, 1)?)?;
                tensor(&mut report, &predicate, tiles)?;
                report.largest_tensor_admitted_bytes = report
                    .largest_tensor_admitted_bytes
                    .max(plan.predicate_workspace_bytes());
                report.scheduled_coordinator_advances =
                    add(report.scheduled_coordinator_advances, tiles)?;
                let metadata = plan.metadata_bytes()?;
                report.compound_plan_metadata_bytes =
                    add(report.compound_plan_metadata_bytes, metadata as u64)?;
                let workspace = plan
                    .peak_extra_bytes()?
                    .checked_sub(metadata)
                    .and_then(|n| n.checked_add(plan.predicate_workspace_bytes()))
                    .ok_or(Error::Bounds)?;
                report.largest_compound_workspace_bytes = report
                    .largest_compound_workspace_bytes
                    .max(workspace as u64);
            }
            Code::Input(_)
            | Code::PublicInput { .. }
            | Code::PublicValues(_)
            | Code::Gather(_)
            | Code::SourceGather(_)
            | Code::Affine { .. } => (),
            Code::Product { .. } => {
                report.scheduled_coordinator_advances =
                    add(report.scheduled_coordinator_advances, 1)?
            }
        }
    }
    report.work_by_code = work_by_code
        .into_iter()
        .map(|(code, work)| (code, work.into()))
        .collect();
    let mut live = vec![false; program.nodes.len()];
    let mut retained = 0;
    for (id, n) in program.nodes.iter().enumerate().take(program.inputs) {
        if matches!(n.code, Code::Input(_)) {
            live[id] = true;
            retained = add(retained, mul(n.lanes as u64, 16)?)?;
            report.largest_secret_owner_values =
                report.largest_secret_owner_values.max(n.lanes as u64);
        }
    }
    report.peak_retained_secret_bytes = retained;
    for (id, n) in program.nodes.iter().enumerate() {
        if id >= program.inputs && !matches!(n.code, Code::PublicInput { .. }) {
            live[id] = true;
            retained = add(retained, mul(n.lanes as u64, 16)?)?;
            report.largest_secret_owner_values =
                report.largest_secret_owner_values.max(n.lanes as u64);
        }
        report.peak_retained_secret_bytes = report.peak_retained_secret_bytes.max(retained);
        for &owner in &program.retire[id] {
            if live[owner] {
                retained = retained
                    .checked_sub(mul(program.nodes[owner].lanes as u64, 16)?)
                    .ok_or(Error::Bounds)?;
                live[owner] = false;
            }
        }
    }
    Ok(report)
}

/// Existing SDK/benchmark inspection hook, separate from source_contract and
/// executable admission. No checkpoint, numerical proof or serving claim.
pub fn common_shape_diagnostic_json(rows: usize, selections: usize) -> Result<Vec<u8>, String> {
    let baseline =
        estimate(Dimensions::COMMON_QWEN, rows, selections).map_err(|e| e.to_string())?;
    let candidate = audit_common_capacity(rows, selections).map_err(|e| e.to_string())?;
    serde_json::to_vec(&serde_json::json!({
        "schema":"pllm.replicated.cost-diagnostic.v1", "scope":"common_qwen_shape_only",
        "executable":false, "admission_evaluated":false,
        "model":"Qwen/Qwen2.5-0.5B-Instruct",
        "revision":"7ae557604adf67be50417f59c2c2f167def9a775",
        "baseline":baseline,"candidate_guard256":candidate
    }))
    .map_err(|e| e.to_string())
}

/// Distinct hook for a real admitted program; never associates that program
/// with the fixed common-Qwen shape inventory without source evidence.
pub fn admitted_diagnostic_json(program: &Program) -> Result<Vec<u8>, String> {
    let counters = admitted(program).map_err(|e| e.to_string())?;
    serde_json::to_vec(&serde_json::json!({
        "schema":"pllm.replicated.cost-diagnostic.v1", "scope":"admitted_native_program",
        "program_digest":program.digest(), "source_descriptor_binding":program.source_binding,
        "counters":counters
    }))
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialized_shape_contract_never_claims_source_admission() {
        let report: serde_json::Value =
            serde_json::from_slice(&common_shape_diagnostic_json(4, 2).unwrap()).unwrap();
        assert_eq!(report["schema"], "pllm.replicated.cost-diagnostic.v1");
        assert_eq!(report["executable"], false);
        assert_eq!(report["admission_evaluated"], false);
        assert!(report.get("counters").is_none());
        assert!(common_shape_diagnostic_json(65, 8).is_err());
    }

    #[test]
    fn dimensions_match_source_locked_metadata_without_loading_checkpoint() {
        let replay: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../paper/artifacts/puma-sdk/local-v4/qwen-metadata-replay.json"
        ))
        .unwrap();
        let config: serde_json::Value =
            serde_json::from_str(replay["normalized_config"].as_str().unwrap()).unwrap();
        let d = Dimensions::COMMON_QWEN;
        for (key, expected) in [
            ("hidden_size", d.hidden),
            ("intermediate_size", d.intermediate),
            ("num_hidden_layers", d.layers),
            ("num_attention_heads", d.heads),
            ("num_key_value_heads", d.kv_heads),
            ("vocab_size", d.vocabulary),
        ] {
            assert_eq!(config[key].as_u64(), Some(expected as u64), "{key}");
        }
        assert_eq!(config["tie_word_embeddings"], d.tied_embeddings);
    }

    #[test]
    fn admitted_tiny_report_reconciles_work_and_retained_owners() {
        use super::super::{Publisher, WeightMode};
        let config = br#"{"model_type":"llama","hidden_size":2,"intermediate_size":2,
            "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
            "vocab_size":3,"max_position_embeddings":64,"rope_theta":10000,
            "rms_norm_eps":0.000001,"tie_word_embeddings":false,"hidden_act":"silu",
            "attention_bias":false,"head_dim":2,"attention_dropout":0.0,"use_cache":true}"#;
        let mut publisher =
            Publisher::new_with_weights(config, 2, 2, [1; 32], [2; 32], WeightMode::Public)
                .unwrap();
        for (key, shape) in publisher.shapes().to_vec() {
            let value = if key.contains("norm") {
                1.0_f32
            } else if key == "model.embed_tokens.weight" {
                1.5
            } else {
                0.0
            };
            let bytes: Vec<_> = (0..shape.iter().product::<usize>())
                .flat_map(|_| value.to_le_bytes())
                .collect();
            publisher.append(&key, 0, &bytes).unwrap();
        }
        publisher.seal(&[0, 1]).unwrap();
        let manifest: super::super::Manifest =
            serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
        let program =
            Program::compile_with_weights(config, 2, 2, &manifest.tensors, WeightMode::Public)
                .unwrap();
        let report = admitted(&program).unwrap();
        assert_eq!(report.work.ordinals, program.operation_limit());
        assert_eq!(
            report.work.aes_blocks_per_party_upper,
            program.work_budget().cursors().aes_work
        );
        assert!(report.tensor_invocations > 0);
        assert!(report.tensor_instructions >= report.tensor_invocations);
        assert!(report.peak_retained_secret_bytes >= 16 * report.largest_secret_owner_values);
        assert!(report.public_ring64_macs_per_worker > 0);
        assert!(report.peer_rounds.is_none());
        assert!(report.peer_frames.is_none());
        assert!(report.peer_bytes.is_none());
        let execution = super::super::ExecutionPolicy {
            range_policy: super::super::RangePolicy::Checked,
            max_abs_activation: 1,
            lookup: super::super::LookupPolicy::Tiled,
            selection: super::super::SelectionPolicy::Tiled,
            ..super::super::ExecutionPolicy::default()
        };
        let checked = Program::compile_with_policy(
            config,
            2,
            2,
            &manifest.tensors,
            super::super::WeightMode::Public,
            execution,
        )
        .unwrap();
        let compound = admitted(&checked).unwrap();
        assert!(compound.nodes_by_code["checked_range"] > 0);
        assert_eq!(compound.nodes_by_code["public_lookup"], 2);
        assert_eq!(compound.nodes_by_code["selection"], 2);
        assert_eq!(compound.stable_selection_comparisons, 4);
        assert_eq!(compound.stable_selection_products, 8);
        assert_eq!(compound.unpriced_native_nodes, 0);
        assert!(compound.tensor_instruction_count_complete);
        assert!(compound.largest_compound_workspace_bytes > 0);
        assert!(compound.compound_plan_metadata_bytes > 0);
        assert_eq!(
            compound
                .work_by_code
                .values()
                .map(|w| w.ordinals)
                .sum::<u64>(),
            checked.operation_limit()
        );
        assert_eq!(
            compound.public_ring64_macs_per_worker,
            report.public_ring64_macs_per_worker
        );
        assert!(
            compound.logical_public_file_read_bytes_per_worker
                < report.logical_public_file_read_bytes_per_worker
        );
        let mut clear = BTreeMap::new();
        for (key, tensor) in &manifest.tensors {
            let value = if key.contains("norm") {
                1 << 18
            } else if key == "model.embed_tokens.weight" {
                3 << 17
            } else {
                0
            };
            clear.insert(key.clone(), vec![value; tensor.shape.iter().product()]);
        }
        clear.insert("input.tokens".into(), vec![0, 1]);
        let program = std::sync::Arc::new(checked);
        let predicted = compound.scheduled_coordinator_advances;
        // Existing admitted three-party native harness, not a benchmark driver.
        let observed = std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(move || {
                let mut workers = super::super::tests::workers(program, &clear);
                let mut advances = 0;
                while !workers.iter().all(super::super::Worker::complete) {
                    advances += 1;
                    assert!(advances <= predicted + 1);
                    let mut queue = Vec::new();
                    for worker in &mut workers {
                        let from = worker.party();
                        queue.extend(
                            worker
                                .advance()
                                .unwrap()
                                .into_iter()
                                .map(|(to, frame)| (from, to, frame)),
                        );
                    }
                    while !queue.is_empty() {
                        let mut next = Vec::new();
                        for (from, to, frame) in queue {
                            next.extend(
                                workers[to as usize]
                                    .receive(frame.as_bytes(), from)
                                    .unwrap()
                                    .into_iter()
                                    .map(|(peer, frame)| (to, peer, frame)),
                            );
                        }
                        queue = next;
                    }
                }
                advances
            })
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(observed, predicted);
    }

    #[test]
    fn common_prefixes_charge_complete_body_head_and_argmax() {
        for (rows, selections, prefixes, lookup_ops) in [
            (4, 2, 9, 299124),
            (16, 2, 33, 1096788),
            (64, 8, 540, 17947440),
        ] {
            let r = estimate(Dimensions::COMMON_QWEN, rows, selections).unwrap();
            assert!(!r.executable);
            assert_eq!(r.recomputed_rows, prefixes);
            assert_eq!(r.categories[0].included_work.ordinals, lookup_ops);
            assert!(r.included_work.ordinals > lookup_ops);
            assert_eq!(
                r.categories
                    .iter()
                    .find(|c| c.name == "rmsnorm")
                    .unwrap()
                    .semantic_invocations,
                49 * prefixes
            );
            assert_eq!(
                r.categories
                    .iter()
                    .find(|c| c.name == "stable_argmax")
                    .unwrap()
                    .output_lanes,
                selections as u64 * 151935
            );
            assert_eq!(
                r.public_ring64_macs_per_worker,
                r.body_public_ring64_macs_per_worker
                    + r.head_public_ring64_macs_per_worker
                    + r.lookup_public_ring64_macs_per_worker
            );
            assert_eq!(
                r.peer_payload_floor.bytes_per_rotation_edge,
                prefixes * 151936 * 112
            );
            assert!(r.categories.iter().any(|c| c.omitted_work.is_some()));
            eprintln!(
                "cost {rows}+{selections}: {}",
                serde_json::to_string(&r).unwrap()
            );
        }
    }

    #[test]
    fn boundary_link_floor_alone_exceeds_lifetime_without_round_confusion() {
        let r = estimate(Dimensions::COMMON_QWEN, 64, 8).unwrap();
        assert!(r.peer_payload_floor.exceeds_1200_seconds_at_40_mbps);
        assert!(r.peer_payload_floor.seconds_at_100_mbps < 1200.0);
        assert_ne!(
            r.peer_payload_floor.lookup_a2b_waves,
            r.categories[0].included_work.ordinals
        );
        assert!(
            r.bulk_logical_public_matrix_read_bytes_per_worker
                < r.logical_public_matrix_read_bytes_per_worker
        );
        assert!(r.included_work.ordinals > super::super::puma::MAX_PROGRAM_OPERATIONS);
    }

    #[test]
    fn invalid_geometry_never_allocates_big_schedules() {
        assert!(estimate(Dimensions::COMMON_QWEN, usize::MAX, 8).is_err());
        assert!(estimate(Dimensions::COMMON_QWEN, 0, 1).is_err());
        assert!(estimate(Dimensions::COMMON_QWEN, 64, 9).is_err());
        let mut d = Dimensions::COMMON_QWEN;
        d.hidden = usize::MAX;
        assert!(estimate(d, 4, 2).is_err());
    }

    #[test]
    fn guard256_dimension_audit_prices_body_nodes_without_changing_baseline() {
        for (rows, selections) in [(4, 2), (16, 2), (64, 8)] {
            let audit = audit_common_capacity(rows, selections).unwrap();
            assert!(!audit.executable);
            assert!(!audit.source_numeric_proof_complete);
            assert_eq!(audit.maximum_guard_tiles, 343 * audit.recomputed_rows);
            let bytes: u64 = audit
                .nodes
                .iter()
                .filter(|n| matches!(n.category, "gate_up_binary" | "residual_binary"))
                .map(|n| n.cumulative_secret_value_bytes)
                .sum();
            assert!(bytes > 64 << 20);
            assert!(audit.priced_unconditional_nodes > 65536);
            assert_eq!(
                audit.compact18_lookup_encoded_bytes_all_directions,
                audit.recomputed_rows * 1187 * 19920
            );
            assert_eq!(
                audit.compact18_a2b_encoded_bytes_per_rotation_edge,
                audit.recomputed_rows * 1187 * 4100
            );
            assert_eq!(
                audit.exceeds_original_coordinator_barrier_budget,
                rows == 64
            );
            assert_eq!(
                audit.silu_sign_payload_bytes_per_rotation_edge_floor,
                24 * audit.recomputed_rows * 4864 * 3 * 112
            );
            if rows == 64 {
                assert!(audit.known_stage_seconds_per_edge_at_100_mbps_floor > 1200.0);
            }
            if let Some(work) = audit.conditional_work_with_all_guards {
                assert!(work.ordinals > 1 << 20);
            }
            eprintln!(
                "capacity {rows}+{selections}: {}",
                serde_json::to_string(&audit).unwrap()
            );
        }
        let baseline = estimate(Dimensions::COMMON_QWEN, 64, 8).unwrap();
        assert_eq!(baseline.profile, PROFILE);
        assert_eq!(baseline.categories[0].included_work.ordinals, 17947440);
        assert_eq!(
            baseline.peer_payload_floor.bytes_per_rotation_edge,
            9189089280
        );
    }
}
