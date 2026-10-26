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

from examples.papers.gkm import experiment as _gkm_experiment


experiment = _gkm_experiment(
    budget=ExecutionBudget(requests=1, max_input_tokens=30, max_new_tokens=2),
).with_params(name="gkm-corrected-prompt")

# Retain the base factory's pinned model, full sharing, decoder capacity,
# ring64/Q11 weights, f32 full client KV, client bias, and private-worker contracts.
