//! Checked link between the existing serial planner and SSA trace validator.

use super::*;

/// External lowering facts absent from the numeric level problem. Kernel order
/// is problem/ledger order, not graph storage order. No profile is selected.
/// These names assert a carrier/key contract; they do not establish key material,
/// backend support, cryptographic legality, or the identity of a hybrid evaluator.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicEncryptedLevelTraceBinding {
    pub input_value: String,
    /// CKKS carrier before and after each complete, atomic hybrid round trip.
    /// No canonical spelling or cryptographic parameter set is inferred.
    pub scheme: String,
    pub key_domain: String,
    pub kernel_identities: Vec<String>,
}

/// Independent existing phase limits, not a combined runtime/solver budget.
/// Both preflights run before solving. The temporary recomputed solution is
/// dropped before trace validation; caller-owned inputs remain borrowed.
/// Linking allocates no additional tables and adds at most quadratic bounded
/// graph-record comparisons plus linear solution/binding comparisons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicEncryptedLevelTraceCapacity {
    pub levels: PublicLevelSolverCapacity,
    pub trace: PublicCiphertextTraceCapacity,
}

/// Consistency validation for these exact immutable borrowed inputs only.
/// Not an executable ciphertext plan, persistent certificate, SDK capability,
/// source-lowering proof, full Budget, physical peak, or security assertion.
/// The trace summary retains the existing manifest-only accounting semantics;
/// its work/buffer fields describe the trace phase, not this entire bridge.
///
/// A live validation prevents mutation of its graph, even via the original owner.
/// ```compile_fail,E0502
/// use pllm_compiler::encrypted_level_schedule::*;
/// fn mutate_graph(
///     p: &PublicLevelProblem, s: &PublicLevelSolution,
///     g: &mut PublicCiphertextGraph, t: &[PublicCiphertextTraceStep],
///     b: &PublicEncryptedLevelTraceBinding, c: PublicEncryptedLevelTraceCapacity,
/// ) {
///     let checked = validate_public_encrypted_level_trace(p, s, g, t, b, c).unwrap();
///     g.operations.clear();
///     assert!(!checked.graph().operations.is_empty());
/// }
/// ```
/// Its explicit lowering binding is borrowed too, not merely copied into a quote.
/// ```compile_fail,E0502
/// use pllm_compiler::encrypted_level_schedule::*;
/// fn mutate_binding(
///     p: &PublicLevelProblem, s: &PublicLevelSolution,
///     g: &PublicCiphertextGraph, t: &[PublicCiphertextTraceStep],
///     b: &mut PublicEncryptedLevelTraceBinding, c: PublicEncryptedLevelTraceCapacity,
/// ) {
///     let checked = validate_public_encrypted_level_trace(p, s, g, t, b, c).unwrap();
///     b.kernel_identities.clear();
///     assert!(!checked.binding().kernel_identities.is_empty());
/// }
/// ```
/// A validation cannot escape a temporary graph's lifetime.
/// ```compile_fail,E0515
/// use pllm_compiler::encrypted_level_schedule::*;
/// fn escape<'a>(
///     p: &'a PublicLevelProblem, s: &'a PublicLevelSolution,
///     g: &PublicCiphertextGraph, t: &'a [PublicCiphertextTraceStep],
///     b: &'a PublicEncryptedLevelTraceBinding, c: PublicEncryptedLevelTraceCapacity,
/// ) -> CheckedPublicEncryptedLevelTrace<'a> {
///     let temporary = g.clone();
///     validate_public_encrypted_level_trace(p, s, &temporary, t, b, c).unwrap()
/// }
/// ```
#[derive(Debug)]
pub struct CheckedPublicEncryptedLevelTrace<'a> {
    problem: &'a PublicLevelProblem,
    solution: &'a PublicLevelSolution,
    graph: &'a PublicCiphertextGraph,
    trace: &'a [PublicCiphertextTraceStep],
    binding: &'a PublicEncryptedLevelTraceBinding,
    summary: PublicCiphertextTraceSummary,
}

impl<'a> CheckedPublicEncryptedLevelTrace<'a> {
    pub fn problem(&self) -> &'a PublicLevelProblem {
        self.problem
    }

    pub fn solution(&self) -> &'a PublicLevelSolution {
        self.solution
    }

    pub fn graph(&self) -> &'a PublicCiphertextGraph {
        self.graph
    }

    pub fn trace(&self) -> &'a [PublicCiphertextTraceStep] {
        self.trace
    }

    pub fn binding(&self) -> &'a PublicEncryptedLevelTraceBinding {
        self.binding
    }

    pub fn trace_summary(&self) -> PublicCiphertextTraceSummary {
        self.summary
    }
}

fn carrier(
    state: &PublicCiphertextState,
    level: u32,
    binding: &PublicEncryptedLevelTraceBinding,
) -> Result<(), String> {
    if state.level != level
        || state.scheme != binding.scheme
        || state.key_domain != binding.key_domain
    {
        return Err("level trace carrier level/scheme/key-domain mismatch".into());
    }
    Ok(())
}

/// Recompute the exact unpruned optimum (including stable table-order ties),
/// compare the entire supplied solution including both digest identities, then
/// validate the supplied trace with the EXISTING typed SSA/liveness gate.
/// Finally require one ordered unary graph operation per ledger operation and
/// exact selected compute/depth/transition or entry/evaluate/recover rows.
/// Equality of totals alone is never sufficient. Views cannot substitute for
/// planned computation, refresh, or recovery, even if their costs coincide.
///
/// Extra immutable input aliases and persistent owners are allowed and remain
/// subject to existing backing/release rules; the computational path must be a
/// single chain with one final output. Branch/join planning is rejected, not
/// flattened. Graph storage order may differ from actual execution order.
///
/// Atomic hybrid rows expose endpoint types and three charge identities, not
/// intermediate ciphertext types. This API checks exactly that existing contract;
/// it cannot establish missing external crypto/key/error assumptions or kernels.
pub fn validate_public_encrypted_level_trace<'a>(
    problem: &'a PublicLevelProblem,
    solution: &'a PublicLevelSolution,
    graph: &'a PublicCiphertextGraph,
    trace: &'a [PublicCiphertextTraceStep],
    binding: &'a PublicEncryptedLevelTraceBinding,
    capacity: PublicEncryptedLevelTraceCapacity,
) -> Result<CheckedPublicEncryptedLevelTrace<'a>, String> {
    preflight(problem, capacity.levels)?;
    trace_preflight(graph, trace, capacity.trace)?;
    if binding.kernel_identities.len() != problem.operations.len()
        || trace.len() != problem.operations.len()
        || graph.maximum_level != problem.maximum_level
    {
        return Err("level trace problem/context mismatch".into());
    }
    for id in [
        &binding.input_value,
        &binding.scheme,
        &binding.key_domain,
        &problem.cost_model_identity,
        &problem.cost_unit,
        &graph.cost_model_identity,
        &graph.cost_unit,
    ]
    .into_iter()
    .chain(&binding.kernel_identities)
    {
        identity(id)?;
    }
    if graph.cost_model_identity != problem.cost_model_identity
        || graph.cost_unit != problem.cost_unit
    {
        return Err("level trace problem/context mismatch".into());
    }
    // Never hash or index the untrusted supplied ledger. Equality against a
    // freshly solved bounded result checks all lengths, indices, charges, levels,
    // identities, totals and hashes, including unused alternatives in the problem.
    // No full problem, graph, trace, or caller solution is cloned.
    {
        let expected = solve_public_encrypted_levels(problem, capacity.levels)?;
        if solution != &expected {
            return Err("level trace solution/problem/digest mismatch".into());
        }
    }
    let summary = validate_public_ciphertext_trace(graph, trace, capacity.trace)?;
    let mut previous = graph
        .inputs
        .iter()
        .find(|v| v.identity == binding.input_value)
        .ok_or("unknown level trace input")?;
    carrier(&previous.state, problem.input_level, binding)?;
    for (i, ((operation, ledger), step)) in problem
        .operations
        .iter()
        .zip(&solution.ledger.steps)
        .zip(trace)
        .enumerate()
    {
        if step.operation != operation.identity {
            return Err("level trace ordered operation mismatch".into());
        }
        let node = graph
            .operations
            .iter()
            .find(|op| op.identity == step.operation)
            .ok_or("unknown level trace operation")?;
        if node.primitive != operation.primitive
            || node.kernel_identity != binding.kernel_identities[i]
            || node.inputs.len() != 1
            || node.inputs[0] != previous.identity
        {
            return Err("level trace operator/kernel/non-chain mismatch".into());
        }
        carrier(&previous.state, ledger.input_level, binding)?;
        carrier(&node.output.state, ledger.output_level, binding)?;
        let capability = graph
            .capabilities
            .iter()
            .find(|cap| cap.identity == step.capability)
            .ok_or("unknown level trace capability")?;
        let exact = match (&capability.execution, ledger.choice) {
            (
                PublicCiphertextExecution::Ckks {
                    compute,
                    depth,
                    transition,
                },
                PublicLevelChoice::Ckks {
                    compute_row,
                    transition_row,
                },
            ) => {
                *depth == operation.depth
                    && operation.compute.get(compute_row) == Some(compute)
                    && operation.transitions.get(transition_row) == Some(transition)
            }
            (
                PublicCiphertextExecution::Hybrid { round_trip },
                PublicLevelChoice::Hybrid { round_trip_row },
            ) => operation.hybrid.get(round_trip_row) == Some(round_trip),
            _ => false,
        };
        if !exact {
            return Err("level trace selected execution/cost/transition mismatch".into());
        }
        previous = &node.output;
    }
    if graph.outputs.len() != 1 || graph.outputs[0] != previous.identity {
        return Err("level trace final output mismatch".into());
    }
    if summary.total_cost != solution.ledger.total_cost {
        return Err("level trace ledger cost conservation failure".into());
    }
    Ok(CheckedPublicEncryptedLevelTrace {
        problem,
        solution,
        graph,
        trace,
        binding,
        summary,
    })
}

#[cfg(test)]
#[path = "encrypted_level_trace_bridge_tests.rs"]
mod tests;
