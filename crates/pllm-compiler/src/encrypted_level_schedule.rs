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
//! Composite/branch/join graphs must first obtain a separately verified lowering;
//! this API rejects non-chain dependencies rather than forgetting live values.
//! Costs are caller-supplied integer units under one public model identity. No
//! measured latency, numeric validity, key ownership or backend support is inferred.

use pllm_types::{canonical_digest, valid_identity, Digest};
use serde::{Deserialize, Serialize};

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

#[cfg(test)]
#[path = "encrypted_level_schedule_tests.rs"]
mod tests;
