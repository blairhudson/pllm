# Reimplementation backlog

What PLLM will reproduce, implement, review, and measure next.

[View canonical HTML](https://pllm.run/research/backlog/)

Document ID: `pllm.docs.research.backlog`  
Release: `0.1.0`  
Build: `sha256:87cbf81718b764da3fb4871c21f12e5943efd717a5d5854e5a0cecf969808585`  
Source hash: `sha256:2a35235ab25647875e7e70eda42ffb73a22aad537b9167a48e2b2580b88e8b2d`

PLLM tracks papers because they may improve a concrete protocol, compiler pass,
numeric primitive, or state-management boundary. A tracked paper is not a PLLM
implementation. A similar data flow is not a reproduction.

## Release-critical work

1. **Complete one executable Qwen profile.** Close every semantic, numeric,
protected, placement, state, and token-feedback gap for the preferred
`research.single_evaluator` profile. The compiler must continue to reject
partial coverage.
2. **Re-measure the current prepared runtime.** Record current-revision Qwen
prefill and decode runs with exact model, prompt length, output length,
machine, topology, cold/warm state, protocol bytes, and failure behavior.
Keep loopback, WAN, CPU, and accelerator claims separate.
3. **Scale and review the experimental gated-MLP components.** The dense-table
baseline and compact mixed-modulus method now share one capability contract;
scalar and independent four-lane schedules are separate selections. Extend
beyond the four-lane bound, validate the resulting memory schedule, and obtain
cryptographic review. Bounded Q7 bundles are not a complete-model security result.

## Composition and search

Papers do not become isolated runtime stacks. Each reimplementation contributes
one or more components to a stable capability family: model and cache
transformations, linear or nonlinear protocols, protected representations,
conversions, preparation schemes, kernels, placement policies, assurance checks,
and benchmark objectives. Like-for-like implementations live together behind
the same typed contract; paper IDs remain provenance, not architecture.
This is an evolving taxonomy. A genuinely new semantic role, lifecycle, or type
boundary can introduce a new capability family; a new paper alone cannot.
Families may split into dedicated modules as their implementation libraries grow
without changing their stable plan contracts.

The harness will explore combinations in stages. It first rejects incompatible
numeric domains, party topologies, trust assumptions, state lifetimes, model
coverage, and devices. It then uses exhaustive or grid search for small bounded
spaces and reproducible seeded random search for larger conditional spaces.
Adaptive search can follow when enough comparable evidence exists. Every
candidate still compiles into an immutable plan and passes correctness, privacy,
execution, and evidence gates before its performance enters a comparison.

Search is multi-objective. Correctness and privacy are hard constraints; latency,
throughput, traffic, memory, preparation cost, quality, energy, and monetary cost
remain separate measurements. PLLM records a Pareto frontier for one matched
cohort rather than declaring one method universally best.

## Paper-driven work

| Priority | Source | PLLM work | Current status |
| --- | --- | --- | --- |
| R24 | Maverick | Specify delegated matrix-vector execution, LPN masking, and batch verification against the prepared public-weight stage contract; independently implement bounded native primitives and run the paper's Qwen3-4B comparison. | Planned; no PLLM implementation or reproduction |
| R23 | MPCache | Validate the existing model-neutral `DecoderPlan` transformation, then implement and measure protected cache selection under an explicitly compatible topology. | Structural plan adaptation only |
| R01-R03 | Dash, ReDash, table-free arithmetic garbling | Finish fidelity, security review, and full decoder-region composition around the existing reference primitives. | R03 source locked and adapted into the compact Q7 implementation with bounded four-lane scheduling; not reproduced |
| R04-R11 | Garbling and decision-program foundations | Implement only the primitives required by an accepted complete-model plan, with paper-scoped tests and view definitions. | Tracked |
| R12-R18 | GPU/MPC preprocessing and nonlinear primitives | Reproduce comparable operators before considering integration; preserve each paper's party count and trust model in reports. | Tracked |
| R19-R22 | Private-transformer systems | Use as full-system comparison targets. Do not transfer their performance or security claims across different models, numeric graphs, or party topologies. | Tracked |

## Required gates

Every reimplementation advances through the same visible gates:

```text
source + version
  -> clean-room specification
  -> inspectable reference
  -> source-scope fidelity tests
  -> reusable native component
  -> typed compiler/runtime integration
  -> scoped security review
  -> matched benchmark
  -> documented promotion decision
```

Missing evidence stays missing. Functional tests do not establish privacy;
operator benchmarks do not establish complete-model execution; loopback runs do
not establish WAN performance; an adaptation does not become a reproduction
without source-scope fidelity evidence.

See [Papers](/research/papers/) for source-by-source summaries and links, the
[PLLM paper](/research/paper/) for the system contract, and
[SDK status](/sdk/reference/status/) for shipped implementation coverage.
