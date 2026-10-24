//! Exact, public, unpruned level/scheme planning; not ciphertext execution or admission.
//!
//! Source facts: pinned CacheMir 2602.11470v1 §6, PDF pp.8–9, Eq.(1),
//! computes before optional bootstrap/drop and requires input >= supplied depth.
//! Its printed dominance rule is deliberately NOT used (depth=1, x=1, y=2).
//! ROSETTA 2609.16915v1 §4, PDF pp.8–10, adds non-arithmetic TFHE choices
//! with entry + evaluation + recovery costs. Its level-zero abstraction does
//! not establish physical StC legality or make conversion free.
//!
//! Chosen method-neutral API: one serial live-value chain, an exhaustive CKKS
//! computation table and explicit post-compute transition capabilities; optional
//! atomic hybrid round trips carry all three cost identities. Absent transition
//! means illegal, NOT unknown/zero cost. Present unknown costs reject globally.
//! The serial API rejects non-chain dependencies rather than forgetting live values.
//! The separate composite trace validator below checks a supplied SSA execution
//! order, exact typed capability matches, immutable parents and backing lifetimes.
//! It does not optimize graphs or certify the caller's cryptographic assumptions.
//! `validate_public_encrypted_level_trace` links those existing contracts: it
//! rechecks the exact solver result and binds its ordered choices to a supplied
//! unary trace, with caller-explicit carrier scheme, key domain and kernels.
//! Costs are caller-supplied integer units under one public model identity. No
//! measured latency, numeric validity, key ownership or backend support is inferred.

use pllm_types::{canonical_digest, valid_identity, Digest};
use serde::{Deserialize, Serialize};

#[path = "encrypted_trace_construction.rs"]
pub mod construction;

#[path = "encrypted_level_trace_bridge.rs"]
mod trace_bridge;
pub use trace_bridge::{
    validate_public_encrypted_level_trace, CheckedPublicEncryptedLevelTrace,
    PublicEncryptedLevelTraceBinding, PublicEncryptedLevelTraceCapacity,
};

pub const ENCRYPTED_LEVEL_PROBLEM_DOMAIN: &str = "pllm.encrypted_level_problem.v1";
pub const ENCRYPTED_LEVEL_LEDGER_DOMAIN: &str = "pllm.encrypted_level_ledger.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicPrimitive {
    Arithmetic,
    Exp,
    Inv,
    Invsqrt,
    Silu,
    Gelu,
    Relu,
    /// An unresolved nonlinear/composite operator is never silently approximated.
    UnknownNonlinear,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCost {
    pub identity: String,
    /// None means unknown, not zero. Zero is allowed only when explicitly supplied.
    pub units: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicLevelCompute {
    pub input_level: u32,
    pub cost: PublicCost,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicLevelTransitionKind {
    Preserve,
    Drop,
    Refresh,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicLevelTransition {
    pub kind: PublicLevelTransitionKind,
    pub from_level: u32,
    pub to_level: u32,
    /// Identity binds the caller's legal transform, not a registered component.
    pub cost: PublicCost,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicHybridRoundTrip {
    pub identity: String,
    pub input_level: u32,
    pub output_level: u32,
    /// Complete legal CKKS-to-evaluator entry, including all required transforms.
    pub entry: PublicCost,
    pub evaluate: PublicCost,
    /// Complete recovery, not raw modulus raising.
    pub recover: PublicCost,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicLevelOperation {
    pub identity: String,
    /// First operation: empty. Every other operation: exactly previous identity.
    pub predecessors: Vec<String>,
    pub primitive: PublicPrimitive,
    pub depth: u32,
    /// Exactly one known entry for every input level depth..=maximum_level.
    /// A depth larger than maximum_level permits hybrid-only evaluation.
    pub compute: Vec<PublicLevelCompute>,
    /// Exhaustive legal post-compute transitions; empty disables all CKKS paths.
    /// Include explicit zero-cost Preserve rows when preservation is legal.
    pub transitions: Vec<PublicLevelTransition>,
    /// Empty disables scheme changes. Each row is a complete legal costed edge.
    pub hybrid: Vec<PublicHybridRoundTrip>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicLevelProblem {
    pub cost_model_identity: String,
    pub cost_unit: String,
    pub maximum_level: u32,
    pub input_level: u32,
    pub output_level: u32,
    pub operations: Vec<PublicLevelOperation>,
}

/// Explicit solver budget, not an encryption/runtime resource policy.
/// Limits reject the WHOLE problem before search, never truncate alternatives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicLevelSolverCapacity {
    pub max_operations: usize,
    pub max_levels: usize,
    pub max_table_rows: usize,
    pub max_vertices: usize,
    /// Bounds validation scans plus complete (including unreachable) edge scans.
    pub max_work: usize,
    /// Conservative storage envelope: 16 KiB per input/output table record plus
    /// DP cells. Includes canonical-JSON digest temporaries, not caller storage
    /// or allocator bookkeeping. Not a measured RSS claim.
    pub max_buffer_bytes: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicLevelChoice {
    Ckks {
        compute_row: usize,
        transition_row: usize,
    },
    Hybrid {
        round_trip_row: usize,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicLevelPathStep {
    pub operation_identity: String,
    pub input_level: u32,
    pub output_level: u32,
    pub choice: PublicLevelChoice,
    /// Ordered compute, then transition; or entry, evaluate, recover.
    pub charges: Vec<PublicCost>,
    pub total_cost: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicLevelCostLedger {
    pub problem_digest: Digest,
    pub steps: Vec<PublicLevelPathStep>,
    pub total_cost: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicLevelSolution {
    pub ledger: PublicLevelCostLedger,
    pub ledger_digest: Digest,
}

fn add(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| "solver size overflow".into())
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| "solver size overflow".into())
}
fn identity(s: &str) -> Result<(), String> {
    if !valid_identity(s) {
        return Err("invalid public identity".into());
    }
    Ok(())
}
fn cost(c: &PublicCost) -> Result<u64, String> {
    identity(&c.identity)?;
    c.units
        .ok_or_else(|| "unknown public cost/capability".into())
}
fn sum_cost(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "public cost overflow".into())
}

/// Preflight uses lengths only before allocations or potentially long scans.
fn preflight(p: &PublicLevelProblem, c: PublicLevelSolverCapacity) -> Result<usize, String> {
    let n = p.operations.len();
    let levels = add(
        usize::try_from(p.maximum_level).map_err(|_| "level overflow")?,
        1,
    )?;
    if n == 0 || n > c.max_operations || levels > c.max_levels {
        return Err("operation/level capacity exceeded".into());
    }
    let vertices = mul(add(n, 1)?, levels)?;
    if vertices > c.max_vertices {
        return Err("vertex capacity exceeded".into());
    }
    let mut rows = n;
    let mut work = vertices;
    for op in &p.operations {
        // Bound dependency vectors even for rejected branch/join input.
        rows = add(rows, op.predecessors.len())?;
        rows = add(
            rows,
            add(
                add(op.compute.len(), op.transitions.len())?,
                op.hybrid.len(),
            )?,
        )?;
        work = add(work, mul(op.compute.len(), op.transitions.len())?)?;
        work = add(work, op.hybrid.len())?;
    }
    // Validation checks uniqueness with bounded pair scans, no hash allocation.
    work = add(work, mul(rows, rows)?)?;
    let bytes = add(
        mul(vertices, std::mem::size_of::<Cell>())?,
        mul(add(rows, mul(n, 5)?)?, 16_384)?,
    )?;
    if rows > c.max_table_rows || work > c.max_work || bytes > c.max_buffer_bytes {
        return Err("table/work/buffer capacity exceeded".into());
    }
    Ok(levels)
}

fn validate(p: &PublicLevelProblem) -> Result<(), String> {
    identity(&p.cost_model_identity)?;
    identity(&p.cost_unit)?;
    if p.input_level > p.maximum_level || p.output_level > p.maximum_level {
        return Err("boundary level outside public range".into());
    }
    for (i, op) in p.operations.iter().enumerate() {
        identity(&op.identity)?;
        if p.operations[..i].iter().any(|x| x.identity == op.identity) {
            return Err("duplicate operation identity".into());
        }
        if (i == 0 && !op.predecessors.is_empty())
            || (i > 0
                && (op.predecessors.len() != 1
                    || op.predecessors[0] != p.operations[i - 1].identity))
        {
            return Err("branch/join or non-chain dependency unsupported".into());
        }
        if op.primitive == PublicPrimitive::UnknownNonlinear {
            return Err("unknown nonlinear primitive".into());
        }
        let expected = if op.depth > p.maximum_level {
            0
        } else {
            u64::from(p.maximum_level) - u64::from(op.depth) + 1
        };
        if u64::try_from(op.compute.len()).map_err(|_| "table size overflow")? != expected {
            return Err("incomplete computation cost table".into());
        }
        for (j, row) in op.compute.iter().enumerate() {
            cost(&row.cost)?;
            if row.input_level < op.depth
                || row.input_level > p.maximum_level
                || op.compute[..j]
                    .iter()
                    .any(|x| x.input_level == row.input_level)
            {
                return Err("invalid/duplicate computation level".into());
            }
        }
        for (j, t) in op.transitions.iter().enumerate() {
            cost(&t.cost)?;
            let legal = match t.kind {
                PublicLevelTransitionKind::Preserve => {
                    t.to_level == t.from_level && t.cost.units == Some(0)
                }
                PublicLevelTransitionKind::Drop => t.to_level < t.from_level,
                PublicLevelTransitionKind::Refresh => t.to_level > t.from_level,
            };
            if !legal
                || t.from_level > p.maximum_level
                || t.to_level > p.maximum_level
                || op.transitions[..j]
                    .iter()
                    .any(|x| x.cost.identity == t.cost.identity)
            {
                return Err("invalid/duplicate legal transition".into());
            }
        }
        for (j, h) in op.hybrid.iter().enumerate() {
            identity(&h.identity)?;
            // Each primitive's complete entry/evaluation/recovery remains explicit.
            sum_cost(
                sum_cost(cost(&h.entry)?, cost(&h.evaluate)?)?,
                cost(&h.recover)?,
            )?;
            if op.primitive == PublicPrimitive::Arithmetic
                || h.input_level > p.maximum_level
                || h.output_level > p.maximum_level
                || op.hybrid[..j].iter().any(|x| x.identity == h.identity)
            {
                return Err("invalid hybrid capability".into());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Cell {
    distance: Option<u64>,
    predecessor: Option<(u32, usize, Option<usize>)>,
}

fn relax(
    cell: &mut Cell,
    base: Option<u64>,
    edge: u64,
    pred: (u32, usize, Option<usize>),
) -> Result<(), String> {
    if let Some(base) = base {
        let candidate = sum_cost(base, edge)?;
        if cell.distance.is_none_or(|old| candidate < old) {
            cell.distance = Some(candidate);
            cell.predecessor = Some(pred);
        }
    }
    Ok(())
}

/// Solve every legal edge in the finite layered DAG, with no dominance pruning.
/// Stable ties keep first encountered predecessor: CKKS before hybrid, then
/// caller table order. Table order is part of the problem digest.
/// Any edge-cost or shortest-prefix arithmetic overflow rejects, even if another
/// path fits. Nonminimal prefixes are not retained at a vertex (ordinary exact
/// shortest-path relaxation, not level/capability dominance pruning).
pub fn solve_public_encrypted_levels(
    p: &PublicLevelProblem,
    capacity: PublicLevelSolverCapacity,
) -> Result<PublicLevelSolution, String> {
    let levels = preflight(p, capacity)?;
    validate(p)?;
    let count = mul(add(p.operations.len(), 1)?, levels)?;
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(count)
        .map_err(|_| "solver allocation failed")?;
    cells.resize(
        count,
        Cell {
            distance: None,
            predecessor: None,
        },
    );
    cells[p.input_level as usize].distance = Some(0);
    for (i, op) in p.operations.iter().enumerate() {
        for (j, row) in op.compute.iter().enumerate() {
            let residual = row.input_level - op.depth;
            let base = cells[i * levels + row.input_level as usize].distance;
            for (k, t) in op.transitions.iter().enumerate() {
                if t.from_level == residual {
                    let edge = sum_cost(cost(&row.cost)?, cost(&t.cost)?)?;
                    relax(
                        &mut cells[(i + 1) * levels + t.to_level as usize],
                        base,
                        edge,
                        (row.input_level, j, Some(k)),
                    )?;
                }
            }
        }
        for (j, h) in op.hybrid.iter().enumerate() {
            let base = cells[i * levels + h.input_level as usize].distance;
            let edge = sum_cost(
                sum_cost(cost(&h.entry)?, cost(&h.evaluate)?)?,
                cost(&h.recover)?,
            )?;
            relax(
                &mut cells[(i + 1) * levels + h.output_level as usize],
                base,
                edge,
                (h.input_level, j, None),
            )?;
        }
    }
    let total_cost = cells[p.operations.len() * levels + p.output_level as usize]
        .distance
        .ok_or("requested output level unreachable")?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(p.operations.len())
        .map_err(|_| "path allocation failed")?;
    let mut output = p.output_level;
    for i in (0..p.operations.len()).rev() {
        let op = &p.operations[i];
        let (input, row, transition) = cells[(i + 1) * levels + output as usize]
            .predecessor
            .ok_or("missing path predecessor")?;
        let (choice, charges) = if let Some(t) = transition {
            (
                PublicLevelChoice::Ckks {
                    compute_row: row,
                    transition_row: t,
                },
                vec![op.compute[row].cost.clone(), op.transitions[t].cost.clone()],
            )
        } else {
            let h = &op.hybrid[row];
            (
                PublicLevelChoice::Hybrid {
                    round_trip_row: row,
                },
                vec![h.entry.clone(), h.evaluate.clone(), h.recover.clone()],
            )
        };
        let mut subtotal = 0;
        for charge in &charges {
            subtotal = sum_cost(subtotal, cost(charge)?)?;
        }
        steps.push(PublicLevelPathStep {
            operation_identity: op.identity.clone(),
            input_level: input,
            output_level: output,
            choice,
            charges,
            total_cost: subtotal,
        });
        output = input;
    }
    steps.reverse();
    let mut conserved = 0;
    for step in &steps {
        conserved = sum_cost(conserved, step.total_cost)?;
    }
    if output != p.input_level || conserved != total_cost {
        return Err("path cost conservation failure".into());
    }
    let ledger = PublicLevelCostLedger {
        problem_digest: canonical_digest(ENCRYPTED_LEVEL_PROBLEM_DOMAIN, p),
        steps,
        total_cost,
    };
    let ledger_digest = canonical_digest(ENCRYPTED_LEVEL_LEDGER_DOMAIN, &ledger);
    Ok(PublicLevelSolution {
        ledger,
        ledger_digest,
    })
}

/// A caller-supplied error bound in one explicitly named model. Known bounds and
/// capability assumptions are required, not inferred or cryptographically proved.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCiphertextError {
    pub model_identity: String,
    pub bound: Option<u64>,
}

/// Physical state plus logical shape. Identities name reviewed external contracts;
/// matching them proves consistency only, never source or semantic provenance.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCiphertextState {
    pub scheme: String,
    pub key_domain: String,
    pub layout: String,
    pub scale: String,
    pub level: u32,
    pub shape: Vec<u64>,
    pub error: PublicCiphertextError,
}

/// Entire backing allocation, not the visible size of a slice/view.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCiphertextOwner {
    pub identity: String,
    pub bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCiphertextValue {
    pub identity: String,
    pub state: PublicCiphertextState,
    pub owner: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PublicCiphertextJoin {
    Unary,
    /// Every input has the same physical state and shape, except error magnitude.
    Aligned,
    /// Exactly two inputs; scalar shape [1], vector shape/layout preserved.
    /// The exact scalar/vector layout mapping must also match the capability.
    ScalarBroadcast {
        vector_input: usize,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PublicCiphertextExecution {
    Ckks {
        compute: PublicLevelCompute,
        depth: u32,
        transition: PublicLevelTransition,
    },
    Hybrid {
        round_trip: PublicHybridRoundTrip,
    },
    /// Read-only logical view of input zero. No modulus/scale/key/layout changes.
    /// The output holds the parent's FULL backing until its own last release.
    View {
        cost: PublicCost,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCiphertextCapability {
    pub identity: String,
    /// Distinguishes, for example, square, reduction, multiply and residual add.
    pub kernel_identity: String,
    pub primitive: PublicPrimitive,
    pub inputs: Vec<PublicCiphertextState>,
    pub output: PublicCiphertextState,
    pub join: PublicCiphertextJoin,
    pub execution: PublicCiphertextExecution,
    /// Mandatory external justification for this exact input/output error mapping.
    pub error_assumption: Option<String>,
    /// Complete temporary owners, including copies, entry/evaluation scratch,
    /// keys/contexts charged to this step, and retained allocator arenas. Zero
    /// must be explicit. Persistent owners belong in the graph owner inventory.
    pub scratch_bytes: Option<u64>,
    /// Additional intermediate/recovery owners, excluding the final output and
    /// scratch above. Conservatively overlaps ALL live input/output backings.
    pub recovery_overlap_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCiphertextGraphOperation {
    pub identity: String,
    pub kernel_identity: String,
    pub primitive: PublicPrimitive,
    pub inputs: Vec<String>,
    pub output: PublicCiphertextValue,
}

/// Source lowering supplies this SSA graph and exact capability manifest. This
/// gate does not establish that a model was faithfully lowered or kernels exist.
/// Operations need not be stored topologically; the trace must execute each once.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCiphertextGraph {
    pub cost_model_identity: String,
    pub cost_unit: String,
    pub maximum_level: u32,
    pub owners: Vec<PublicCiphertextOwner>,
    /// Keys, contexts and other persistent owners with no ciphertext handle.
    pub persistent_owners: Vec<String>,
    pub inputs: Vec<PublicCiphertextValue>,
    pub operations: Vec<PublicCiphertextGraphOperation>,
    pub capabilities: Vec<PublicCiphertextCapability>,
    pub outputs: Vec<String>,
    /// Input handles that may not be retired (e.g. retained KV parents).
    pub retained_inputs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCiphertextTraceStep {
    pub operation: String,
    pub capability: String,
    /// Explicit handle releases AFTER output/scratch/recovery overlap is charged.
    pub release: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicCiphertextTraceCapacity {
    /// Includes nested state/shape/reference records, not just operation count.
    pub max_records: usize,
    pub max_work: usize,
    /// Validator metadata payload envelope; not ciphertext allocation or RSS.
    /// Caller-owned graph/trace storage and allocator bookkeeping are external.
    pub max_buffer_bytes: usize,
    /// All inventoried live backings plus simultaneous scratch/recovery owners.
    pub max_live_owner_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicCiphertextTraceSummary {
    pub total_cost: u64,
    pub peak_live_backing_bytes: u64,
    pub peak_live_owner_bytes: u64,
    pub final_live_backing_bytes: u64,
    pub work_bound: usize,
    pub buffer_bytes: usize,
}

fn trace_records(n: &mut usize, extra: usize, limit: usize) -> Result<(), String> {
    *n = add(*n, extra)?;
    if *n > limit {
        return Err("composite record capacity exceeded".into());
    }
    Ok(())
}

/// Shared record-to-validator envelope. Construction quotes and legacy graph
/// preflight deliberately use the same checked formula and unchanged limits.
fn trace_resource_bound(records: usize) -> Option<(usize, usize)> {
    let work = records
        .checked_mul(records)?
        .checked_mul(records)?
        .checked_mul(64)?
        .checked_add(records.checked_mul(256)?)?;
    let bytes = records.checked_mul(256)?.checked_add(1024)?;
    Some((work, bytes))
}

/// Record accounting shared by borrowed preconstruction and legacy graph walk.
struct TraceRecordCounter {
    records: usize,
}
impl TraceRecordCounter {
    fn new() -> Self {
        Self { records: 1 }
    }
    fn top(&mut self, n: usize) -> Option<()> {
        self.records = self.records.checked_add(n)?;
        Some(())
    }
    fn state(&mut self, rank: usize) -> Option<()> {
        self.top(9usize.checked_add(rank)?)
    }
    fn operation(&mut self, arity: usize) -> Option<()> {
        self.top(3usize.checked_add(arity)?)
    }
    fn capability(&mut self, arity: usize) -> Option<()> {
        self.top(12usize.checked_add(arity)?)
    }
    fn step(&mut self, releases: usize) -> Option<()> {
        self.top(2usize.checked_add(releases)?)
    }
}

/// Length-only, allocation-free preflight. Bound nested loops before entering
/// them. A deliberately conservative cubic bound covers pair scans, typed
/// comparisons and complete reference/lifetime scans; there is no state search.
fn trace_preflight(
    g: &PublicCiphertextGraph,
    trace: &[PublicCiphertextTraceStep],
    c: PublicCiphertextTraceCapacity,
) -> Result<(usize, usize), String> {
    let mut counter = TraceRecordCounter::new();
    for len in [
        g.owners.len(),
        g.persistent_owners.len(),
        g.inputs.len(),
        g.operations.len(),
        g.capabilities.len(),
        g.outputs.len(),
        g.retained_inputs.len(),
        trace.len(),
    ] {
        counter.top(len).ok_or("composite resource overflow")?;
        trace_records(&mut 0, counter.records, c.max_records)?;
    }
    let state_records = |n: &mut TraceRecordCounter, s: &PublicCiphertextState| {
        n.state(s.shape.len())
            .ok_or("composite resource overflow")?;
        trace_records(&mut 0, n.records, c.max_records)
    };
    for v in &g.inputs {
        state_records(&mut counter, &v.state)?;
    }
    for op in &g.operations {
        counter
            .operation(op.inputs.len())
            .ok_or("composite resource overflow")?;
        state_records(&mut counter, &op.output.state)?;
    }
    for cap in &g.capabilities {
        counter
            .capability(cap.inputs.len())
            .ok_or("composite resource overflow")?;
        state_records(&mut counter, &cap.output)?;
        for s in &cap.inputs {
            state_records(&mut counter, s)?;
        }
    }
    for step in trace {
        counter
            .step(step.release.len())
            .ok_or("composite resource overflow")?;
        trace_records(&mut 0, counter.records, c.max_records)?;
    }
    let (work, bytes) =
        trace_resource_bound(counter.records).ok_or("composite resource overflow")?;
    // Six bounded vectors, borrowed values only; no state/String cloning or
    // canonical serialization. Leave ample payload room for Vec capacities.
    if work > c.max_work || bytes > c.max_buffer_bytes {
        return Err("composite work/buffer capacity exceeded".into());
    }
    Ok((work, bytes))
}

fn trace_state(s: &PublicCiphertextState, maximum: u32) -> Result<(), String> {
    for id in [
        &s.scheme,
        &s.key_domain,
        &s.layout,
        &s.scale,
        &s.error.model_identity,
    ] {
        identity(id)?;
    }
    if s.level > maximum || s.shape.is_empty() || s.error.bound.is_none() {
        return Err("unknown error or invalid composite state".into());
    }
    let mut elements = 1u64;
    for &dim in &s.shape {
        if dim == 0 {
            return Err("zero composite dimension".into());
        }
        elements = elements
            .checked_mul(dim)
            .ok_or("composite shape overflow")?;
    }
    Ok(())
}

fn same_domain(a: &PublicCiphertextState, b: &PublicCiphertextState) -> bool {
    a.scheme == b.scheme
        && a.key_domain == b.key_domain
        && a.scale == b.scale
        && a.level == b.level
        && a.error.model_identity == b.error.model_identity
}

fn same_backing_state(a: &PublicCiphertextState, b: &PublicCiphertextState) -> bool {
    same_domain(a, b) && a.layout == b.layout && a.error == b.error
}

fn trace_capability(cap: &PublicCiphertextCapability, maximum: u32) -> Result<u64, String> {
    identity(&cap.identity)?;
    identity(&cap.kernel_identity)?;
    identity(
        cap.error_assumption
            .as_deref()
            .ok_or("unknown error assumption")?,
    )?;
    cap.scratch_bytes.ok_or("unknown scratch owners")?;
    cap.recovery_overlap_bytes
        .ok_or("unknown recovery owners")?;
    sum_cost(
        cap.scratch_bytes.unwrap(),
        cap.recovery_overlap_bytes.unwrap(),
    )?;
    if cap.inputs.is_empty() || cap.primitive == PublicPrimitive::UnknownNonlinear {
        return Err("unknown primitive or missing composite inputs".into());
    }
    for s in cap.inputs.iter().chain(std::iter::once(&cap.output)) {
        trace_state(s, maximum)?;
    }
    let first = &cap.inputs[0];
    match cap.join {
        PublicCiphertextJoin::Unary if cap.inputs.len() == 1 => (),
        PublicCiphertextJoin::Aligned if cap.inputs.len() >= 2 => {
            if cap.inputs.iter().any(|s| {
                !same_domain(first, s) || first.layout != s.layout || first.shape != s.shape
            }) || cap.output.layout != first.layout
                || cap.output.shape != first.shape
            {
                return Err("incompatible aligned join".into());
            }
        }
        PublicCiphertextJoin::ScalarBroadcast { vector_input }
            if cap.inputs.len() == 2 && vector_input < 2 =>
        {
            let vector = &cap.inputs[vector_input];
            let scalar = &cap.inputs[1 - vector_input];
            if !same_domain(vector, scalar)
                || scalar.shape.as_slice() != [1]
                || cap.output.shape != vector.shape
                || cap.output.layout != vector.layout
            {
                return Err("incompatible scalar broadcast join".into());
            }
        }
        _ => return Err("invalid composite join arity".into()),
    }
    match &cap.execution {
        PublicCiphertextExecution::Ckks {
            compute,
            depth,
            transition: t,
        } => {
            let legal = match t.kind {
                PublicLevelTransitionKind::Preserve => {
                    t.to_level == t.from_level && t.cost.units == Some(0)
                }
                PublicLevelTransitionKind::Drop => t.to_level < t.from_level,
                PublicLevelTransitionKind::Refresh => t.to_level > t.from_level,
            };
            if !legal
                || compute.input_level < *depth
                || compute.input_level > maximum
                || t.from_level != compute.input_level - *depth
                || t.to_level != cap.output.level
                || t.to_level > maximum
                || cap.inputs.iter().any(|s| {
                    s.level != compute.input_level
                        || s.scheme != cap.output.scheme
                        || s.key_domain != cap.output.key_domain
                })
            {
                return Err("illegal composite compute/transition".into());
            }
            sum_cost(cost(&compute.cost)?, cost(&t.cost)?)
        }
        PublicCiphertextExecution::Hybrid { round_trip: h } => {
            identity(&h.identity)?;
            if cap.primitive == PublicPrimitive::Arithmetic
                || cap.inputs.len() != 1
                || h.input_level != first.level
                || h.output_level != cap.output.level
            {
                return Err("illegal composite hybrid entry/recovery".into());
            }
            sum_cost(
                sum_cost(cost(&h.entry)?, cost(&h.evaluate)?)?,
                cost(&h.recover)?,
            )
        }
        PublicCiphertextExecution::View { cost: charge } => {
            if cap.primitive != PublicPrimitive::Arithmetic
                || cap.inputs.len() != 1
                || !same_backing_state(first, &cap.output)
            {
                return Err("view cannot mutate physical parent state".into());
            }
            let elements = |s: &PublicCiphertextState| s.shape.iter().product::<u64>();
            if elements(&cap.output) > elements(first) {
                return Err("view exceeds parent extent".into());
            }
            cost(charge)
        }
    }
}

fn trace_vec<T: Clone>(count: usize, value: T) -> Result<Vec<T>, String> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| "composite allocation failed")?;
    result.resize(count, value);
    Ok(result)
}

/// Validate a supplied execution trace, not an optimum. All inputs are immutable;
/// every non-view output needs a fresh owner, even if its input dies immediately.
/// Values remain live until explicit release. Views retain whole backing owners.
/// Output allocation, scratch and recovery overlap are charged BEFORE releases.
///
/// Known declarations (including UNUSED capabilities) must be internally legal.
/// This checks an external capability manifest, not its physics/security proof,
/// actual kernel implementation, complete host RSS, or source-model equivalence.
pub fn validate_public_ciphertext_trace(
    g: &PublicCiphertextGraph,
    trace: &[PublicCiphertextTraceStep],
    capacity: PublicCiphertextTraceCapacity,
) -> Result<PublicCiphertextTraceSummary, String> {
    let (work_bound, buffer_bytes) = trace_preflight(g, trace, capacity)?;
    identity(&g.cost_model_identity)?;
    identity(&g.cost_unit)?;
    if g.inputs.is_empty()
        || g.operations.is_empty()
        || g.outputs.is_empty()
        || trace.len() != g.operations.len()
    {
        return Err("incomplete composite graph/trace".into());
    }
    // Validate bounded-length identifiers before any pair/reference comparisons.
    for owner in &g.owners {
        identity(&owner.identity)?;
        if owner.bytes.is_none_or(|b| b == 0) {
            return Err("unknown or empty backing owner".into());
        }
    }
    for value in g
        .inputs
        .iter()
        .chain(g.operations.iter().map(|op| &op.output))
    {
        identity(&value.identity)?;
        identity(&value.owner)?;
        trace_state(&value.state, g.maximum_level)?;
    }
    for op in &g.operations {
        identity(&op.identity)?;
        identity(&op.kernel_identity)?;
        if op.inputs.is_empty() || op.primitive == PublicPrimitive::UnknownNonlinear {
            return Err("unknown composite operation".into());
        }
        for id in &op.inputs {
            identity(id)?;
        }
    }
    for id in g
        .outputs
        .iter()
        .chain(&g.retained_inputs)
        .chain(&g.persistent_owners)
    {
        identity(id)?;
    }
    for step in trace {
        identity(&step.operation)?;
        identity(&step.capability)?;
        for id in &step.release {
            identity(id)?;
        }
    }
    for (i, cap) in g.capabilities.iter().enumerate() {
        trace_capability(cap, g.maximum_level)?;
        if g.capabilities[..i]
            .iter()
            .any(|c| c.identity == cap.identity)
        {
            return Err("duplicate composite capability".into());
        }
    }
    for (i, op) in g.operations.iter().enumerate() {
        if g.operations[..i].iter().any(|x| x.identity == op.identity) {
            return Err("duplicate composite operation".into());
        }
    }
    for (i, owner) in g.owners.iter().enumerate() {
        if g.owners[..i].iter().any(|x| x.identity == owner.identity) {
            return Err("duplicate composite owner".into());
        }
    }
    for (i, input) in g.inputs.iter().enumerate() {
        if g.inputs[..i].iter().any(|prior| {
            prior.owner == input.owner && !same_backing_state(&prior.state, &input.state)
        }) {
            return Err("aliased inputs disagree on physical backing state".into());
        }
    }
    let count = add(g.inputs.len(), g.operations.len())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| "composite allocation failed")?;
    values.extend(g.inputs.iter());
    values.extend(g.operations.iter().map(|op| &op.output));
    let value_index = |id: &str| {
        values
            .iter()
            .position(|v| v.identity == id)
            .ok_or("unknown composite value")
    };
    let mut owners = trace_vec(count, 0usize)?;
    for (i, value) in values.iter().enumerate() {
        if values[..i].iter().any(|v| v.identity == value.identity) {
            return Err("duplicate composite value".into());
        }
        owners[i] = g
            .owners
            .iter()
            .position(|o| o.identity == value.owner)
            .ok_or("unknown backing owner")?;
    }
    for owner in &g.owners {
        if !values.iter().any(|v| v.owner == owner.identity)
            && !g.persistent_owners.contains(&owner.identity)
        {
            return Err("unused backing owner".into());
        }
    }
    for op in &g.operations {
        for id in &op.inputs {
            value_index(id)?;
        }
    }
    for (i, id) in g.outputs.iter().enumerate() {
        value_index(id)?;
        if g.outputs[..i].contains(id) {
            return Err("duplicate composite output".into());
        }
    }
    for (i, id) in g.retained_inputs.iter().enumerate() {
        if !g.inputs.iter().any(|v| &v.identity == id) || g.retained_inputs[..i].contains(id) {
            return Err("invalid retained parent".into());
        }
    }
    let mut live = trace_vec(count, false)?;
    let mut executed = trace_vec(g.operations.len(), false)?;
    let mut references = trace_vec(g.owners.len(), 0usize)?;
    let mut created = trace_vec(g.owners.len(), false)?;
    let mut backing = 0u64;
    for (i, id) in g.persistent_owners.iter().enumerate() {
        let owner = g
            .owners
            .iter()
            .position(|o| &o.identity == id)
            .ok_or("unknown persistent owner")?;
        if g.persistent_owners[..i].contains(id) {
            return Err("duplicate persistent owner".into());
        }
        references[owner] = 1;
        created[owner] = true;
        backing = sum_cost(backing, g.owners[owner].bytes.unwrap())?;
    }
    for i in 0..g.inputs.len() {
        live[i] = true;
        if references[owners[i]] == 0 {
            backing = sum_cost(backing, g.owners[owners[i]].bytes.unwrap())?;
        }
        references[owners[i]] = add(references[owners[i]], 1)?;
        created[owners[i]] = true;
    }
    let mut peak_backing = backing;
    let mut peak_owners = backing;
    if backing > capacity.max_live_owner_bytes {
        return Err("composite owner capacity exceeded".into());
    }
    let mut total_cost = 0;
    for step in trace {
        let index = g
            .operations
            .iter()
            .position(|op| op.identity == step.operation)
            .ok_or("unknown trace operation")?;
        if executed[index] {
            return Err("duplicate trace execution".into());
        }
        let op = &g.operations[index];
        let cap = g
            .capabilities
            .iter()
            .find(|c| c.identity == step.capability)
            .ok_or("unknown trace capability")?;
        if cap.kernel_identity != op.kernel_identity
            || cap.primitive != op.primitive
            || cap.inputs.len() != op.inputs.len()
            || cap.output != op.output.state
        {
            return Err("composite capability/output mismatch".into());
        }
        for (id, expected) in op.inputs.iter().zip(&cap.inputs) {
            let input = value_index(id)?;
            if !live[input] || values[input].state != *expected {
                return Err("unavailable parent (cycle, early release or type mismatch)".into());
            }
        }
        let output = g.inputs.len() + index;
        let owner = owners[output];
        if matches!(&cap.execution, PublicCiphertextExecution::View { .. }) {
            if owner != owners[value_index(&op.inputs[0])?] {
                return Err("view must retain parent backing".into());
            }
        } else {
            // Fresh means never owned by an input or another non-view producer,
            // not merely currently dead. No hidden in-place/cache mutation.
            if created[owner] {
                return Err("non-view output needs fresh immutable backing".into());
            }
            created[owner] = true;
        }
        if references[owner] == 0 {
            backing = sum_cost(backing, g.owners[owner].bytes.unwrap())?;
        }
        references[owner] = add(references[owner], 1)?;
        live[output] = true;
        executed[index] = true;
        let overlap = sum_cost(
            sum_cost(backing, cap.scratch_bytes.unwrap())?,
            cap.recovery_overlap_bytes.unwrap(),
        )?;
        peak_backing = peak_backing.max(backing);
        peak_owners = peak_owners.max(overlap);
        if overlap > capacity.max_live_owner_bytes {
            return Err("composite owner capacity exceeded".into());
        }
        total_cost = sum_cost(total_cost, trace_capability(cap, g.maximum_level)?)?;
        for id in &step.release {
            let released = value_index(id)?;
            if !live[released] || g.retained_inputs.contains(id) {
                return Err("invalid release or immutable retained parent retirement".into());
            }
            live[released] = false;
            references[owners[released]] -= 1;
            if references[owners[released]] == 0 {
                backing -= g.owners[owners[released]].bytes.unwrap();
            }
        }
    }
    for id in g.outputs.iter().chain(&g.retained_inputs) {
        if !live[value_index(id)?] {
            return Err("missing composite boundary value".into());
        }
    }
    Ok(PublicCiphertextTraceSummary {
        total_cost,
        peak_live_backing_bytes: peak_backing,
        peak_live_owner_bytes: peak_owners,
        final_live_backing_bytes: backing,
        work_bound,
        buffer_bytes,
    })
}

#[cfg(test)]
#[path = "encrypted_level_schedule_tests.rs"]
mod tests;

/// Exact bounded planning over the existing fixed-state SSA ciphertext graph.
#[path = "encrypted_level_dag.rs"]
pub mod dag;
