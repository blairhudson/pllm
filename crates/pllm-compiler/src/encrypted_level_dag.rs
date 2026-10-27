//! Exact finite planning over the existing typed SSA graph, not HE execution.
//!
//! The serial level problem cannot express forked live values. This minimal DAG
//! contract uses `PublicCiphertextGraph` directly: source lowering fixes every
//! value's state, backing owner and ordered parents. Search chooses a topological
//! execution order and an exact existing capability for each operation. All
//! matching CKKS/hybrid/view alternatives are retained, including costlier rows
//! with smaller declared scratch/recovery overlap. No level, scheme, key, error,
//! conversion, cost or owner is invented; unspecified variants are NOT searched.
//!
//! Exhaustive prefix-frontier search extends the serial planner's exact finite-
//! path search principle, replacing its single-live-value DP cell with a complete
//! execution prefix. Costs reuse the same checked helpers and typed rows. No
//! unsafe call to that chain-only solver, dominance pruning, owner merging or
//! DAG flattening occurs. Every complete candidate is checked by the existing
//! SSA trace validator. After each step, dead non-boundary handles are released
//! eagerly: delaying a dead handle cannot reduce declared owner overlap, and
//! fresh-owner rules prohibit storage reuse even after release. This is exact
//! within that fixed graph/manifest and existing post-step-release trace model.
//!
//! The owner ceiling is a declared-manifest feasibility condition, NOT measured
//! backend capacity, host RSS or a full Budget. All existing external crypto/key/
//! error assumptions remain assumptions. No component or capability is registered.

use super::*;

pub const ENCRYPTED_DAG_PROBLEM_DOMAIN: &str = "pllm.encrypted_level_dag_problem.v1";
pub const ENCRYPTED_DAG_LEDGER_DOMAIN: &str = "pllm.encrypted_level_dag_ledger.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicEncryptedDagCapacity {
    pub max_operations: usize,
    /// Includes the root and EVERY generated type-compatible prefix, without
    /// deduplication, even if complete owner validation later rejects that path.
    pub max_states: usize,
    /// Maximum simultaneously retained prefixes; exhaustion rejects the whole
    /// search, including any incumbent. This is not beam-search truncation.
    pub max_frontier: usize,
    /// One conservative work quantum for static validation, each expanded
    /// prefix, and finalization. Quantum = 128*R^3 + 16384*R, where R counts the
    /// graph plus an upper bound on generated trace/reference records below.
    /// Units are a metadata scan envelope, never elapsed time or crypto cost.
    pub max_work: usize,
    /// Conservative metadata/serialization payload envelope. Caller storage and
    /// allocator bookkeeping are external; this is not physical memory admission.
    pub max_buffer_bytes: usize,
    /// Existing trace-phase limits. Live-owner bytes define path feasibility;
    /// exhaustion of other trace limits rejects the whole search.
    pub trace: PublicCiphertextTraceCapacity,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicEncryptedDagSelection {
    pub operation_index: usize,
    pub capability_index: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicEncryptedDagLedger {
    /// Binds ALL ordered graph declarations, including unused capabilities and
    /// exact input/output levels, schemes, domains, costs and owner identities.
    pub problem_digest: Digest,
    pub selections: Vec<PublicEncryptedDagSelection>,
    pub trace: Vec<PublicCiphertextTraceStep>,
    pub total_cost: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicEncryptedDagSearchSummary {
    pub states_generated: usize,
    pub states_expanded: usize,
    pub peak_frontier: usize,
    pub work_quantum: usize,
    pub work: usize,
    pub buffer_bytes: usize,
}

/// Search accounting and existing trace accounting are separate metadata
/// envelopes. Neither summary establishes measured memory or execution capacity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicEncryptedDagSolution {
    pub ledger: PublicEncryptedDagLedger,
    pub ledger_digest: Digest,
    pub trace_summary: PublicCiphertextTraceSummary,
    pub search_summary: PublicEncryptedDagSearchSummary,
}

/// A consistency check cannot outlive or permit mutation of its borrowed source.
/// No executable ciphertext capability is produced.
///
/// ```compile_fail,E0502
/// use pllm_compiler::encrypted_level_schedule::{dag::*, PublicCiphertextGraph};
/// fn mutate(g: &mut PublicCiphertextGraph, s: &PublicEncryptedDagSolution,
///           c: PublicEncryptedDagCapacity) {
///     let checked = validate_public_encrypted_dag_solution(g, s, c).unwrap();
///     g.owners.clear();
///     assert!(!checked.graph().owners.is_empty());
/// }
/// ```
/// ```compile_fail,E0515
/// use pllm_compiler::encrypted_level_schedule::{dag::*, PublicCiphertextGraph};
/// fn escape<'a>(g: &PublicCiphertextGraph, s: &'a PublicEncryptedDagSolution,
///               c: PublicEncryptedDagCapacity) -> CheckedPublicEncryptedDag<'a> {
///     let temporary = g.clone();
///     validate_public_encrypted_dag_solution(&temporary, s, c).unwrap()
/// }
/// ```
#[derive(Debug)]
pub struct CheckedPublicEncryptedDag<'a> {
    graph: &'a PublicCiphertextGraph,
    solution: &'a PublicEncryptedDagSolution,
}

impl<'a> CheckedPublicEncryptedDag<'a> {
    pub fn graph(&self) -> &'a PublicCiphertextGraph {
        self.graph
    }

    pub fn solution(&self) -> &'a PublicEncryptedDagSolution {
        self.solution
    }
}

fn work_quantum(records: usize) -> Result<usize, String> {
    add(
        mul(mul(mul(records, records)?, records)?, 128)?,
        mul(records, 16_384)?,
    )
}

fn charge_work(work: &mut usize, quantum: usize, limit: usize) -> Result<(), String> {
    *work = add(*work, quantum)?;
    if *work > limit {
        return Err("DAG work capacity exceeded; optimum not established".into());
    }
    Ok(())
}

/// Length-only preflight, before identifier scans, allocations or serialization.
fn preflight_dag(
    g: &PublicCiphertextGraph,
    c: PublicEncryptedDagCapacity,
) -> Result<(usize, usize), String> {
    let n = g.operations.len();
    if n == 0 || n > c.max_operations || c.max_states == 0 || c.max_frontier == 0 {
        return Err("DAG operation/state/frontier capacity exceeded".into());
    }
    // Same records and bounded scans as the existing trace preflight, including
    // nested shapes. Add a conservative generated trace: n step records, 2*n
    // step fields, and at most one release per input/produced handle.
    let mut records = 1;
    let limit = c.trace.max_records;
    for len in [
        g.owners.len(),
        g.persistent_owners.len(),
        g.inputs.len(),
        n,
        g.capabilities.len(),
        g.outputs.len(),
        g.retained_inputs.len(),
        mul(n, 3)?,
        add(g.inputs.len(), n)?,
    ] {
        trace_records(&mut records, len, limit)?;
    }
    for value in &g.inputs {
        trace_records(&mut records, add(9, value.state.shape.len())?, limit)?;
    }
    for op in &g.operations {
        trace_records(&mut records, add(3, op.inputs.len())?, limit)?;
        trace_records(&mut records, add(9, op.output.state.shape.len())?, limit)?;
    }
    for cap in &g.capabilities {
        trace_records(&mut records, add(12, cap.inputs.len())?, limit)?;
        trace_records(&mut records, add(9, cap.output.shape.len())?, limit)?;
        for input in &cap.inputs {
            trace_records(&mut records, add(9, input.shape.len())?, limit)?;
        }
    }
    let quantum = work_quantum(records)?;
    // Strings are bounded identities; shape elements count separately. Covers
    // borrowed metadata, trace/ledger clones, canonical JSON temporaries, and
    // a full n-element path per frontier cell plus two work/best paths.
    let bytes = add(
        mul(add(records, add(mul(n, 6)?, 8)?)?, 16_384)?,
        mul(mul(add(c.max_frontier, 2)?, add(n, 1)?)?, 128)?,
    )?;
    if bytes > c.max_buffer_bytes {
        return Err("DAG buffer capacity exceeded".into());
    }
    Ok((quantum, bytes))
}

fn source_value<'a>(g: &'a PublicCiphertextGraph, id: &str) -> Option<&'a PublicCiphertextValue> {
    g.inputs
        .iter()
        .chain(g.operations.iter().map(|op| &op.output))
        .find(|v| v.identity == id)
}

fn declarations(g: &PublicCiphertextGraph) -> Result<(), String> {
    identity(&g.cost_model_identity)?;
    identity(&g.cost_unit)?;
    let mut all_backings = 0;
    for owner in &g.owners {
        identity(&owner.identity)?;
        let bytes = owner
            .bytes
            .filter(|&b| b > 0)
            .ok_or("unknown or empty backing owner")?;
        all_backings = sum_cost(all_backings, bytes)?;
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
    for cap in &g.capabilities {
        trace_capability(cap, g.maximum_level)?;
        // Reject even a dormant arithmetic overflow globally. This conservative
        // inventory envelope may reject mutually non-live huge backings; it is
        // NOT a claim that they must all be physically resident together.
        sum_cost(
            sum_cost(all_backings, cap.scratch_bytes.unwrap())?,
            cap.recovery_overlap_bytes.unwrap(),
        )?;
    }
    Ok(())
}

pub(crate) fn matches(
    g: &PublicCiphertextGraph,
    op: &PublicCiphertextGraphOperation,
    cap: &PublicCiphertextCapability,
) -> bool {
    cap.kernel_identity == op.kernel_identity
        && cap.primitive == op.primitive
        && cap.output == op.output.state
        && cap.inputs.len() == op.inputs.len()
        && op
            .inputs
            .iter()
            .zip(&cap.inputs)
            .all(|(id, expected)| source_value(g, id).is_some_and(|v| v.state == *expected))
}

fn done(path: &[PublicEncryptedDagSelection], index: usize) -> bool {
    path.iter().any(|p| p.operation_index == index)
}

fn ready(
    g: &PublicCiphertextGraph,
    path: &[PublicEncryptedDagSelection],
    op: &PublicCiphertextGraphOperation,
) -> bool {
    op.inputs.iter().all(|id| {
        g.inputs.iter().any(|v| v.identity == *id)
            || g.operations
                .iter()
                .position(|p| p.output.identity == *id)
                .is_some_and(|i| done(path, i))
    })
}

fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), String> {
    values
        .try_reserve_exact(1)
        .map_err(|_| "DAG allocation failed")?;
    values.push(value);
    Ok(())
}

fn make_trace(
    g: &PublicCiphertextGraph,
    path: &[PublicEncryptedDagSelection],
) -> Result<Vec<PublicCiphertextTraceStep>, String> {
    let mut trace: Vec<PublicCiphertextTraceStep> = Vec::new();
    for (position, selected) in path.iter().enumerate() {
        let op = &g.operations[selected.operation_index];
        let cap = &g.capabilities[selected.capability_index];
        let mut release = Vec::new();
        for value in g
            .inputs
            .iter()
            .chain(g.operations.iter().map(|p| &p.output))
        {
            let id = &value.identity;
            let available = g.inputs.iter().any(|v| &v.identity == id)
                || path[..=position]
                    .iter()
                    .any(|p| g.operations[p.operation_index].output.identity == *id);
            if available
                && !g.outputs.contains(id)
                && !g.retained_inputs.contains(id)
                && !trace.iter().any(|s| s.release.contains(id))
                && !path[position + 1..]
                    .iter()
                    .any(|p| g.operations[p.operation_index].inputs.contains(id))
            {
                push(&mut release, id.clone())?;
            }
        }
        push(
            &mut trace,
            PublicCiphertextTraceStep {
                operation: op.identity.clone(),
                capability: cap.identity.clone(),
                release,
            },
        )?;
    }
    Ok(trace)
}

struct Prefix {
    path: Vec<PublicEncryptedDagSelection>,
    cost: u64,
}

/// Return an exact optimum, or an explicit failure, never a truncated incumbent.
/// Ties retain the lexicographically first sequence of (graph operation index,
/// capability index). All table order, states, domains, costs and owners bind the
/// problem digest. Original graph storage is borrowed, never cloned.
pub fn solve_public_encrypted_dag(
    g: &PublicCiphertextGraph,
    capacity: PublicEncryptedDagCapacity,
) -> Result<PublicEncryptedDagSolution, String> {
    let (quantum, buffer_bytes) = preflight_dag(g, capacity)?;
    let mut work = 0;
    charge_work(&mut work, quantum, capacity.max_work)?;
    declarations(g)?;
    let mut frontier = Vec::new();
    // Reserve the bounded queue once: per-push reallocation could otherwise
    // copy an exponentially large frontier outside a prefix's cubic work unit.
    // Vec storage is uninitialized; allocator bookkeeping is outside the stated
    // scan envelope. Payload reservation is covered by preflight's byte bound.
    frontier
        .try_reserve_exact(capacity.max_frontier.min(capacity.max_states))
        .map_err(|_| "DAG allocation failed")?;
    push(
        &mut frontier,
        Prefix {
            path: Vec::new(),
            cost: 0,
        },
    )?;
    let mut generated = 1;
    let mut expanded = 0;
    let mut peak_frontier = 1;
    let mut best: Option<(
        Prefix,
        Vec<PublicCiphertextTraceStep>,
        PublicCiphertextTraceSummary,
    )> = None;
    while let Some(state) = frontier.pop() {
        charge_work(&mut work, quantum, capacity.max_work)?;
        expanded = add(expanded, 1)?;
        if state.path.len() == g.operations.len() {
            let trace = make_trace(g, &state.path)?;
            // Metadata/work limits may NEVER silently discard a candidate.
            trace_preflight(g, &trace, capacity.trace)?;
            match validate_public_ciphertext_trace(g, &trace, capacity.trace) {
                Ok(summary) => {
                    if summary.total_cost != state.cost {
                        return Err("DAG cost conservation failure".into());
                    }
                    if best
                        .as_ref()
                        .is_none_or(|(old, _, _)| state.cost < old.cost)
                    {
                        best = Some((state, trace, summary));
                    }
                }
                // These are candidate-specific feasibility failures in the
                // existing String error contract. All other errors are fatal,
                // notably allocation, arithmetic, metadata and source errors.
                Err(error)
                    if matches!(
                        error.as_str(),
                        "composite owner capacity exceeded"
                            | "view must retain parent backing"
                            | "non-view output needs fresh immutable backing"
                    ) => {}
                Err(error) => return Err(error),
            }
            continue;
        }
        // Reverse pushes implement forward lexicographic DFS without pruning.
        for (index, op) in g.operations.iter().enumerate().rev() {
            if done(&state.path, index) || !ready(g, &state.path, op) {
                continue;
            }
            for (cap_index, cap) in g.capabilities.iter().enumerate().rev() {
                if !matches(g, op, cap) {
                    continue;
                }
                let cost = sum_cost(state.cost, trace_capability(cap, g.maximum_level)?)?;
                generated = add(generated, 1)?;
                if generated > capacity.max_states {
                    return Err("DAG state capacity exceeded; optimum not established".into());
                }
                if add(frontier.len(), 1)? > capacity.max_frontier {
                    return Err("DAG frontier capacity exceeded; optimum not established".into());
                }
                let mut path = Vec::new();
                path.try_reserve_exact(g.operations.len())
                    .map_err(|_| "DAG allocation failed")?;
                path.extend_from_slice(&state.path);
                path.push(PublicEncryptedDagSelection {
                    operation_index: index,
                    capability_index: cap_index,
                });
                push(&mut frontier, Prefix { path, cost })?;
                peak_frontier = peak_frontier.max(frontier.len());
            }
        }
    }
    let (winner, trace, trace_summary) =
        best.ok_or("no feasible typed DAG trace within declared owner ceiling")?;
    charge_work(&mut work, quantum, capacity.max_work)?;
    let ledger = PublicEncryptedDagLedger {
        problem_digest: canonical_digest(ENCRYPTED_DAG_PROBLEM_DOMAIN, g),
        selections: winner.path,
        trace,
        total_cost: winner.cost,
    };
    Ok(PublicEncryptedDagSolution {
        ledger_digest: canonical_digest(ENCRYPTED_DAG_LEDGER_DOMAIN, &ledger),
        ledger,
        trace_summary,
        search_summary: PublicEncryptedDagSearchSummary {
            states_generated: generated,
            states_expanded: expanded,
            peak_frontier,
            work_quantum: quantum,
            work,
            buffer_bytes,
        },
    })
}

/// Re-solve and compare the ENTIRE supplied result, not merely totals/hashes.
/// Same declared search ceilings are required (metadata envelope is included).
/// Exact node/capability order, all typed execution rows and release/backing
/// accounting are checked through the existing trace machinery. Private borrowed
/// result is consistency validation only, not executable crypto or SDK admission.
pub fn validate_public_encrypted_dag_solution<'a>(
    g: &'a PublicCiphertextGraph,
    solution: &'a PublicEncryptedDagSolution,
    capacity: PublicEncryptedDagCapacity,
) -> Result<CheckedPublicEncryptedDag<'a>, String> {
    if *solution != solve_public_encrypted_dag(g, capacity)? {
        return Err("DAG solution/problem/ledger/trace mismatch".into());
    }
    Ok(CheckedPublicEncryptedDag { graph: g, solution })
}

#[cfg(test)]
#[path = "encrypted_level_dag_tests.rs"]
mod tests;

/// Opt-in tie objective; the original exhaustive entrypoint is unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PublicEncryptedDagMemoObjective {
    #[default]
    CostThenLexicographic,
    CostThenPeakThenLexicographic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicEncryptedDagMemoCapacity {
    /// Total admitted labels, INCLUDING retired labels. No eviction or recycling
    /// hides work or permits a truncated optimum. Multiple Pareto labels may
    /// have the same completed-node set.
    pub max_entries: usize,
    pub objective: PublicEncryptedDagMemoObjective,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicEncryptedDagMemoResources {
    pub path_slots: usize,
    pub key_words: usize,
    pub frontier_slots: usize,
    pub arena_bytes: usize,
    pub metadata_bytes: usize,
    pub buffer_bytes: usize,
    pub work_quantum: usize,
    /// Initialization AND cleanup of all exact slots, static validation and
    /// finalization. Charged in full before the first allocation factory.
    pub minimum_work: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicEncryptedDagMemoSummary {
    pub entries: usize,
    /// Rejected candidate extensions, not the retired labels still in entries.
    pub dominated: usize,
    pub owner_infeasible: usize,
    /// Literal arena-label visits (including retired/other-depth labels).
    pub probes: usize,
    pub resources: PublicEncryptedDagMemoResources,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicEncryptedDagMemoSolution {
    pub solution: PublicEncryptedDagSolution,
    pub memo: PublicEncryptedDagMemoSummary,
}

#[derive(Clone, Copy)]
struct MemoLabel {
    len: usize,
    cost: u64,
    peak: u64,
    active: bool,
}

struct MemoArena {
    labels: Box<[MemoLabel]>,
    paths: Box<[PublicEncryptedDagSelection]>,
    keys: Box<[usize]>,
    frontier: Box<[usize]>,
    path: Box<[PublicEncryptedDagSelection]>,
    key: Box<[usize]>,
}

#[derive(Default)]
struct MemoCounters {
    entries: usize,
    frontier: usize,
    generated: usize,
    expanded: usize,
    peak_frontier: usize,
    dominated: usize,
    owner_infeasible: usize,
    probes: usize,
    work: usize,
    best: Option<usize>,
}

#[cfg(test)]
thread_local! {
    static MEMO_FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn memo_box<T: Copy>(count: usize, value: T) -> Result<Box<[T]>, String> {
    #[cfg(test)]
    MEMO_FACTORIES.with(|n| n.set(n.get() + 1));
    crate::bounded_index::exact_box(count, |_| value)
        .map_err(|_| "DAG memo exact factory/layout failure".into())
}

fn memo_layout<T>(count: usize) -> Result<usize, String> {
    crate::bounded_index::layout_bytes::<T>(count).map_err(|_| "DAG memo layout overflow".into())
}

// The caller has already quoted every exact layout and charged initialization /
// cleanup work. This private factory owns no graph/string/crypto backing.
fn memo_arena(
    n: usize,
    entries: usize,
    r: PublicEncryptedDagMemoResources,
) -> Result<MemoArena, String> {
    let empty = PublicEncryptedDagSelection {
        operation_index: 0,
        capability_index: 0,
    };
    let vacant = MemoLabel {
        len: 0,
        cost: 0,
        peak: 0,
        active: false,
    };
    Ok(MemoArena {
        labels: memo_box(entries, vacant)?,
        paths: memo_box(r.path_slots, empty)?,
        keys: memo_box(r.key_words, 0usize)?,
        frontier: memo_box(r.frontier_slots, 0usize)?,
        path: memo_box(n, empty)?,
        key: memo_box(r.key_words / entries, 0usize)?,
    })
}

/// Allocation-free, length/layout-only quote. Existing metadata/serialization
/// allowance is retained ONCE; its old frontier/work/best-path allowance is
/// replaced, not also credited to this memo. Exact layouts cover six arenas,
/// their inline owner/control records and the final owned selection array.
/// Nested graph/trace/JSON owners remain in the separate metadata allowance.
/// Caller inputs, allocator headers, physical call frames and crypto are external.
pub fn quote_public_encrypted_dag_memo(
    g: &PublicCiphertextGraph,
    c: PublicEncryptedDagCapacity,
    m: PublicEncryptedDagMemoCapacity,
) -> Result<PublicEncryptedDagMemoResources, String> {
    use std::mem::size_of;
    if m.max_entries == 0 {
        return Err("DAG memo entry capacity exceeded".into());
    }
    let mut lengths = c;
    lengths.max_buffer_bytes = usize::MAX;
    let (base_quantum, old_bytes) = preflight_dag(g, lengths)?;
    let n = g.operations.len();
    let words = add(n, usize::BITS as usize - 1)? / usize::BITS as usize;
    let path_slots = mul(m.max_entries, n)?;
    let key_words = mul(m.max_entries, words)?;
    let frontier_slots = c.max_frontier.min(c.max_states);
    let mut arena_bytes = 0;
    for bytes in [
        memo_layout::<MemoLabel>(m.max_entries)?,
        memo_layout::<PublicEncryptedDagSelection>(path_slots)?,
        memo_layout::<usize>(key_words)?,
        memo_layout::<usize>(frontier_slots)?,
        memo_layout::<PublicEncryptedDagSelection>(n)?,
        memo_layout::<usize>(words)?,
        memo_layout::<PublicEncryptedDagSelection>(n)?, // returned ledger
        size_of::<MemoArena>(),
        size_of::<MemoCounters>(),
        mul(2, size_of::<MemoLabel>())?, // root/candidate transition locals
        size_of::<PublicEncryptedDagMemoResources>(),
        size_of::<PublicEncryptedDagMemoCapacity>(),
        size_of::<PublicEncryptedDagMemoSolution>(), // returned inline wrapper
    ] {
        arena_bytes = add(arena_bytes, bytes)?;
    }
    let old_paths = mul(mul(add(c.max_frontier, 2)?, add(n, 1)?)?, 128)?;
    let metadata_bytes = old_bytes
        .checked_sub(old_paths)
        .ok_or("DAG quote mismatch")?;
    let buffer_bytes = add(metadata_bytes, arena_bytes)?;
    if buffer_bytes > c.max_buffer_bytes {
        return Err("DAG memo buffer capacity exceeded".into());
    }
    // Each expansion/transition pays this conservative source-scan envelope,
    // including repeated typed validation. Memo probes, key/path comparisons,
    // copying and initialization/cleanup pay ADDITIONAL literal scalar units.
    let work_quantum = mul(base_quantum, 8)?;
    let slots = add(
        add(
            add(m.max_entries, path_slots)?,
            add(key_words, frontier_slots)?,
        )?,
        add(n, words)?,
    )?;
    let minimum_work = add(mul(2, slots)?, mul(2, work_quantum)?)?;
    Ok(PublicEncryptedDagMemoResources {
        path_slots,
        key_words,
        frontier_slots,
        arena_bytes,
        metadata_bytes,
        buffer_bytes,
        work_quantum,
        minimum_work,
    })
}

/// Source-only checks needed BEFORE treating completed-node sets as equivalent.
/// Reuses the existing state/capability validators; no crypto legality is added.
/// In particular, duplicate identifiers, missing references and incompatible
/// input aliases cannot disappear behind a memo hit. The final trace still goes
/// through the authoritative existing trace validator.
fn memo_source(g: &PublicCiphertextGraph) -> Result<(), String> {
    declarations(g)?;
    if g.inputs.is_empty() || g.outputs.is_empty() {
        return Err("incomplete composite graph/trace".into());
    }
    for (i, cap) in g.capabilities.iter().enumerate() {
        if g.capabilities[..i]
            .iter()
            .any(|c| c.identity == cap.identity)
        {
            return Err("duplicate composite capability".into());
        }
    }
    for (i, owner) in g.owners.iter().enumerate() {
        if g.owners[..i].iter().any(|o| o.identity == owner.identity) {
            return Err("duplicate composite owner".into());
        }
        if !g
            .inputs
            .iter()
            .chain(g.operations.iter().map(|o| &o.output))
            .any(|v| v.owner == owner.identity)
            && !g.persistent_owners.contains(&owner.identity)
        {
            return Err("unused backing owner".into());
        }
    }
    for (i, input) in g.inputs.iter().enumerate() {
        if g.inputs[..i]
            .iter()
            .any(|p| p.owner == input.owner && !same_backing_state(&p.state, &input.state))
        {
            return Err("aliased inputs disagree on physical backing state".into());
        }
    }
    for (i, value) in g
        .inputs
        .iter()
        .chain(g.operations.iter().map(|o| &o.output))
        .enumerate()
    {
        if g.inputs
            .iter()
            .chain(g.operations.iter().map(|o| &o.output))
            .take(i)
            .any(|v| v.identity == value.identity)
        {
            return Err("duplicate composite value".into());
        }
        if !g.owners.iter().any(|o| o.identity == value.owner) {
            return Err("unknown backing owner".into());
        }
    }
    // Nonnegative fixed row costs: sum of per-operation maxima detects even a
    // dormant typed-path overflow before dominance can discard the expensive row.
    // This is a conservative static inventory contract of the OPT-IN API.
    let mut maximum_total = 0;
    for (i, op) in g.operations.iter().enumerate() {
        if g.operations[..i].iter().any(|o| o.identity == op.identity) {
            return Err("duplicate composite operation".into());
        }
        if op.inputs.is_empty() || op.primitive == PublicPrimitive::UnknownNonlinear {
            return Err("unknown composite operation".into());
        }
        for id in &op.inputs {
            source_value(g, id).ok_or("unknown composite value")?;
        }
        let mut maximum = None;
        for cap in &g.capabilities {
            if matches(g, op, cap) {
                let cost = trace_capability(cap, g.maximum_level)?;
                maximum = Some(maximum.map_or(cost, |old: u64| old.max(cost)));
            }
        }
        maximum_total = sum_cost(
            maximum_total,
            maximum.ok_or("no feasible typed DAG trace within declared owner ceiling")?,
        )?;
    }
    for (i, id) in g.outputs.iter().enumerate() {
        source_value(g, id).ok_or("unknown composite value")?;
        if g.outputs[..i].contains(id) {
            return Err("duplicate composite output".into());
        }
    }
    for (i, id) in g.retained_inputs.iter().enumerate() {
        if !g.inputs.iter().any(|v| v.identity == *id) || g.retained_inputs[..i].contains(id) {
            return Err("invalid retained parent".into());
        }
    }
    for (i, id) in g.persistent_owners.iter().enumerate() {
        if !g.owners.iter().any(|o| o.identity == *id) {
            return Err("unknown persistent owner".into());
        }
        if g.persistent_owners[..i].contains(id) {
            return Err("duplicate persistent owner".into());
        }
    }
    Ok(())
}

fn memo_done(key: &[usize], index: usize) -> bool {
    key[index / usize::BITS as usize] & (1usize << (index % usize::BITS as usize)) != 0
}

fn memo_live(
    g: &PublicCiphertextGraph,
    key: &[usize],
    len: usize,
    value: &PublicCiphertextValue,
) -> bool {
    let available = g.inputs.iter().any(|v| v.identity == value.identity)
        || g.operations
            .iter()
            .enumerate()
            .any(|(i, o)| o.output.identity == value.identity && memo_done(key, i));
    available
        && (len == 0
            || g.outputs.contains(&value.identity)
            || g.retained_inputs.contains(&value.identity)
            || g.operations
                .iter()
                .enumerate()
                .any(|(i, o)| !memo_done(key, i) && o.inputs.contains(&value.identity)))
}

fn memo_owner_live(g: &PublicCiphertextGraph, key: &[usize], len: usize, owner: &str) -> bool {
    g.persistent_owners.iter().any(|o| o == owner)
        || g.inputs
            .iter()
            .chain(g.operations.iter().map(|o| &o.output))
            .any(|v| v.owner == owner && memo_live(g, key, len, v))
}

fn memo_backing(g: &PublicCiphertextGraph, key: &[usize], len: usize) -> Result<u64, String> {
    let mut bytes = 0;
    for owner in &g.owners {
        if memo_owner_live(g, key, len, &owner.identity) {
            bytes = sum_cost(bytes, owner.bytes.unwrap())?;
        }
    }
    Ok(bytes)
}

fn memo_transition(
    g: &PublicCiphertextGraph,
    key: &[usize],
    old: MemoLabel,
    index: usize,
    cap_index: usize,
    ceiling: u64,
) -> Result<Option<u64>, String> {
    let op = &g.operations[index];
    let cap = &g.capabilities[cap_index];
    let owner = &op.output.owner;
    if matches!(&cap.execution, PublicCiphertextExecution::View { .. }) {
        if source_value(g, &op.inputs[0]).unwrap().owner != *owner {
            return Ok(None);
        }
    } else if g.persistent_owners.contains(owner)
        || g.inputs.iter().any(|v| v.owner == *owner)
        || g.operations
            .iter()
            .enumerate()
            .any(|(i, o)| memo_done(key, i) && o.output.owner == *owner)
    {
        return Ok(None);
    }
    let mut backing = memo_backing(g, key, old.len)?;
    if !memo_owner_live(g, key, old.len, owner) {
        backing = sum_cost(
            backing,
            g.owners
                .iter()
                .find(|o| o.identity == *owner)
                .unwrap()
                .bytes
                .unwrap(),
        )?;
    }
    let peak = old.peak.max(sum_cost(
        sum_cost(backing, cap.scratch_bytes.unwrap())?,
        cap.recovery_overlap_bytes.unwrap(),
    )?);
    Ok((peak <= ceiling).then_some(peak))
}

fn memo_lex(
    a: &[PublicEncryptedDagSelection],
    b: &[PublicEncryptedDagSelection],
) -> std::cmp::Ordering {
    a.iter()
        .map(|p| (p.operation_index, p.capability_index))
        .cmp(b.iter().map(|p| (p.operation_index, p.capability_index)))
}

fn memo_dominates(
    a: MemoLabel,
    ap: &[PublicEncryptedDagSelection],
    b: MemoLabel,
    bp: &[PublicEncryptedDagSelection],
    objective: PublicEncryptedDagMemoObjective,
) -> bool {
    a.cost < b.cost
        || (a.cost == b.cost
            && (objective == PublicEncryptedDagMemoObjective::CostThenLexicographic
                || a.peak <= b.peak)
            && !memo_lex(ap, bp).is_gt())
}

/// Exact opt-in memoization, not a beam or a backend execution capability.
///
/// Invariant (one borrowed immutable source per solve): a completed-node set D
/// fixes all available value identities and their FULL typed states/backings.
/// Eager release fixes live handles from D plus remaining consumer references;
/// alias counts and persistent backing refs follow, not just a byte total. Every
/// produced view aliases an already-created parent, so ever-created owners are
/// exactly inputs/persistent owners plus owners of D. Capability choices cannot
/// change those fixed future states. Static source checks establish uniqueness
/// and references before any memo hit. No key is shared across source graphs.
///
/// For equal D, the same suffix is legal and has the same additional cost and
/// future overlap. Lower prefix cost strictly wins (all costs are provided u64,
/// globally checked for overflow). At equal cost, legacy ties need lex <= only;
/// peak-aware ties require BOTH peak <= AND lex <=. A lower peak with later lex
/// alone cannot dominate: a suffix can equalize peaks. Thus both Pareto labels
/// survive. The returned complete trace is rechecked by the existing validator.
///
/// max_states counts every attempted typed extension INCLUDING memo/owner rejects;
/// frontier/entry/work exhaustion rejects the WHOLE search, never its incumbent.
/// Linear arena probes and full key/path spans are explicitly metered; arena
/// capacity is fixed before factories. Original solver/result/error contract is
/// unchanged. Memo identifiers/paths contain indices only, never String clones.
pub fn solve_public_encrypted_dag_memo(
    g: &PublicCiphertextGraph,
    c: PublicEncryptedDagCapacity,
    m: PublicEncryptedDagMemoCapacity,
) -> Result<PublicEncryptedDagMemoSolution, String> {
    let r = quote_public_encrypted_dag_memo(g, c, m)?;
    let n = g.operations.len();
    let words = r.key_words / m.max_entries;
    let mut s = MemoCounters::default();
    charge_work(&mut s.work, r.minimum_work, c.max_work)?;
    memo_source(g)?;
    let mut a = memo_arena(n, m.max_entries, r)?;
    let initial = memo_backing(g, &a.key, 0)?;
    if initial > c.trace.max_live_owner_bytes {
        return Err("no feasible typed DAG trace within declared owner ceiling".into());
    }
    a.labels[0] = MemoLabel {
        active: true,
        peak: initial,
        ..a.labels[0]
    };
    s.entries = 1;
    s.frontier = 1;
    s.generated = 1;
    s.peak_frontier = 1;
    while s.frontier != 0 {
        charge_work(&mut s.work, r.work_quantum, c.max_work)?;
        s.frontier -= 1;
        let parent_index = a.frontier[s.frontier];
        let old = a.labels[parent_index];
        if !old.active {
            continue;
        }
        s.expanded = add(s.expanded, 1)?;
        if old.len == n {
            let old_path = &a.paths[parent_index * n..parent_index * n + old.len];
            let trace = make_trace(g, old_path)?;
            trace_preflight(g, &trace, c.trace)?;
            let checked = validate_public_ciphertext_trace(g, &trace, c.trace)?;
            if checked.total_cost != old.cost || checked.peak_live_owner_bytes != old.peak {
                return Err("DAG memo cost/owner conservation failure".into());
            }
            let better = s.best.is_none_or(|best| {
                let previous = a.labels[best];
                old.cost
                    .cmp(&previous.cost)
                    .then_with(|| {
                        if m.objective
                            == PublicEncryptedDagMemoObjective::CostThenPeakThenLexicographic
                        {
                            old.peak.cmp(&previous.peak)
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .then_with(|| memo_lex(old_path, &a.paths[best * n..best * n + n]))
                    .is_lt()
            });
            if better {
                s.best = Some(parent_index);
            }
            continue;
        }
        for (index, op) in g.operations.iter().enumerate().rev() {
            if done(
                &a.paths[parent_index * n..parent_index * n + old.len],
                index,
            ) || !ready(
                g,
                &a.paths[parent_index * n..parent_index * n + old.len],
                op,
            ) {
                continue;
            }
            for (cap_index, cap) in g.capabilities.iter().enumerate().rev() {
                if !matches(g, op, cap) {
                    continue;
                }
                s.generated = add(s.generated, 1)?;
                if s.generated > c.max_states {
                    return Err("DAG state capacity exceeded; optimum not established".into());
                }
                charge_work(&mut s.work, r.work_quantum, c.max_work)?;
                let parent_key = &a.keys[parent_index * words..(parent_index + 1) * words];
                let Some(peak) = memo_transition(
                    g,
                    parent_key,
                    old,
                    index,
                    cap_index,
                    c.trace.max_live_owner_bytes,
                )?
                else {
                    s.owner_infeasible = add(s.owner_infeasible, 1)?;
                    continue;
                };
                let candidate = MemoLabel {
                    len: old.len + 1,
                    cost: sum_cost(old.cost, trace_capability(cap, g.maximum_level)?)?,
                    peak,
                    active: true,
                };
                charge_work(&mut s.work, add(candidate.len, words)?, c.max_work)?;
                a.path[..old.len]
                    .copy_from_slice(&a.paths[parent_index * n..parent_index * n + old.len]);
                a.path[old.len] = PublicEncryptedDagSelection {
                    operation_index: index,
                    capability_index: cap_index,
                };
                a.key.copy_from_slice(parent_key);
                a.key[index / usize::BITS as usize] |= 1usize << (index % usize::BITS as usize);
                let mut dominated = false;
                for entry in 0..s.entries {
                    charge_work(&mut s.work, 1, c.max_work)?;
                    s.probes = add(s.probes, 1)?;
                    let previous = a.labels[entry];
                    if !previous.active || previous.len != candidate.len {
                        continue;
                    }
                    charge_work(&mut s.work, add(words, candidate.len)?, c.max_work)?;
                    if &a.keys[entry * words..(entry + 1) * words] == a.key.as_ref()
                        && memo_dominates(
                            previous,
                            &a.paths[entry * n..entry * n + candidate.len],
                            candidate,
                            &a.path[..candidate.len],
                            m.objective,
                        )
                    {
                        dominated = true;
                        break;
                    }
                }
                if dominated {
                    s.dominated = add(s.dominated, 1)?;
                    continue;
                }
                if s.entries == m.max_entries {
                    return Err("DAG memo entry capacity exceeded; optimum not established".into());
                }
                if s.frontier == r.frontier_slots {
                    return Err("DAG frontier capacity exceeded; optimum not established".into());
                }
                for entry in 0..s.entries {
                    charge_work(&mut s.work, 1, c.max_work)?;
                    s.probes = add(s.probes, 1)?;
                    let previous = a.labels[entry];
                    if !previous.active || previous.len != candidate.len {
                        continue;
                    }
                    charge_work(&mut s.work, add(words, candidate.len)?, c.max_work)?;
                    if &a.keys[entry * words..(entry + 1) * words] == a.key.as_ref()
                        && memo_dominates(
                            candidate,
                            &a.path[..candidate.len],
                            previous,
                            &a.paths[entry * n..entry * n + candidate.len],
                            m.objective,
                        )
                    {
                        a.labels[entry].active = false;
                    }
                }
                charge_work(&mut s.work, add(candidate.len, words)?, c.max_work)?;
                let entry = s.entries;
                a.labels[entry] = candidate;
                a.paths[entry * n..entry * n + candidate.len]
                    .copy_from_slice(&a.path[..candidate.len]);
                a.keys[entry * words..(entry + 1) * words].copy_from_slice(&a.key);
                a.frontier[s.frontier] = entry;
                s.entries += 1;
                s.frontier += 1;
                s.peak_frontier = s.peak_frontier.max(s.frontier);
            }
        }
    }
    let best = s
        .best
        .ok_or("no feasible typed DAG trace within declared owner ceiling")?;
    let winner = a.labels[best];
    let path = &a.paths[best * n..best * n + n];
    let trace = make_trace(g, path)?;
    let trace_summary = validate_public_ciphertext_trace(g, &trace, c.trace)?;
    let mut selections = Vec::new();
    selections
        .try_reserve_exact(n)
        .map_err(|_| "DAG allocation failed")?;
    if selections.capacity() != n {
        return Err("DAG selection capacity mismatch".into());
    }
    selections.extend_from_slice(path);
    let ledger = PublicEncryptedDagLedger {
        problem_digest: canonical_digest(ENCRYPTED_DAG_PROBLEM_DOMAIN, g),
        selections,
        trace,
        total_cost: winner.cost,
    };
    Ok(PublicEncryptedDagMemoSolution {
        solution: PublicEncryptedDagSolution {
            ledger_digest: canonical_digest(ENCRYPTED_DAG_LEDGER_DOMAIN, &ledger),
            ledger,
            trace_summary,
            search_summary: PublicEncryptedDagSearchSummary {
                states_generated: s.generated,
                states_expanded: s.expanded,
                peak_frontier: s.peak_frontier,
                work_quantum: r.work_quantum,
                work: s.work,
                buffer_bytes: r.buffer_bytes,
            },
        },
        memo: PublicEncryptedDagMemoSummary {
            entries: s.entries,
            dominated: s.dominated,
            owner_infeasible: s.owner_infeasible,
            probes: s.probes,
            resources: r,
        },
    })
}

/// Recompute the opt-in result, including objective, accounting and all ledgers.
pub fn validate_public_encrypted_dag_memo_solution<'a>(
    g: &'a PublicCiphertextGraph,
    solution: &'a PublicEncryptedDagMemoSolution,
    c: PublicEncryptedDagCapacity,
    m: PublicEncryptedDagMemoCapacity,
) -> Result<CheckedPublicEncryptedDag<'a>, String> {
    if *solution != solve_public_encrypted_dag_memo(g, c, m)? {
        return Err("DAG memo solution/problem/ledger/trace mismatch".into());
    }
    Ok(CheckedPublicEncryptedDag {
        graph: g,
        solution: &solution.solution,
    })
}

#[cfg(test)]
#[path = "encrypted_level_dag_memo_tests.rs"]
mod memo_tests;
