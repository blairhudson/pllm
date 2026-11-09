"""Ordinary GKM benchmark configuration for corrected ``Hello`` rendering.

Intended request: 30 source-rendered input tokens and two greedy selections
(30 prefill positions plus one feedback position), not the legacy three-token
cohort. Source-owned rendering and ordinary admission remain authoritative;
a different count must be reviewed through admission, never forced by prompt
tampering, hardcoded token vectors, public token receipts, or fallback.
Historical cohorts are not comparable. This configuration is not execution
authorization or numerical-admission, fit, security, or publication evidence.
It collects no source bodies and does not run a benchmark on import.
"""

from pllm import ExecutionBudget
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario

from examples.papers.gkm import experiment as _gkm_experiment


experiment = _gkm_experiment(
    budget=ExecutionBudget(requests=1, max_input_tokens=30, max_new_tokens=2),
).with_params(name="gkm-corrected-prompt")

# Retain the base factory's pinned model, full sharing, decoder capacity,
# ring64/Q11 weights, f32 full client KV, client bias, and private-worker contracts.


def suite(*, memory_bytes, selected=experiment):
    """One cold measured Hello request; caller supplies ordinary admission budget.

    Persistence of salted audit bindings is not independent numeric equivalence.
    This definition grants no execution or memory-allowance authority.
    """
    target = BenchmarkTarget.response(selected, scenario=Scenario.cold(max_output_tokens=2),
                                      backend="native", state_audit=True)
    return BenchmarkSuite([BenchmarkCase("gkm-corrected-prompt", target)],
                          Sampling(processes=1, warmups=0, repetitions=1,
                                   timeout_seconds=450, memory_bytes=memory_bytes))
