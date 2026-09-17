# Reimplementation backlog

What PLLM will reproduce, implement, review, and measure next.

[View canonical HTML](https://pllm.run/research/backlog/)

Document ID: `pllm.docs.research.backlog`  
Release: `0.1.0`  
Build: `sha256:426652b6512bb11e794ef7caf6e150d2b19f0c9f7b933d041295b3bacebb441a`  
Source hash: `sha256:e8d34ef74488bece6645a7ce3418cc444a80f873a353d035f82c3f294ec696fc`

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
3. **Tensorize and review the experimental gated-MLP component.** Replace the
scalar binary projection with a bounded tensor method, validate its Q7 numeric
schedule, one-use lifecycle, authenticated plan binding, and evaluator view,
then obtain cryptographic review. The current scalar composition is not a
complete-model security result.

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
| R01-R03 | Dash, ReDash, table-free arithmetic garbling | Finish fidelity, security review, and full decoder-region composition around the existing reference primitives. | Reference primitives only; not reproduced |
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
