# Architecture

## Product boundary

PLLM resolves a pinned model, numerical graph, privacy contract, role topology, workload, and
device policy into a validated role-sliced execution plan. It is a compiler/runtime and evidence
system, not a protocol switchboard. Unsupported operations MUST fail; no plaintext, Python,
alternate protected-computation, extra-party, or client-heavy fallback may silently change the
selected contract.

## Object lifecycle

```text
Config -> LogicalPlan -> ExecutionPlan -> PlanLock
                                      |
                                      v
                     Runtime -> Session -> PreparedMaterial
```

- `Config` is immutable, JSON-safe public intent. Python constructors, YAML, and CLI overrides
  MUST produce the same canonical configuration and digest.
- `LogicalPlan` freezes model semantics, numeric semantics, protected representations, explicit
  conversions, role policy, workload bounds, and unresolved hardware-independent choices.
- `ExecutionPlan` resolves every method, conversion, kernel, role slice, buffer, message,
  preparation request, and target capability. It contains no `auto` choices.
- `PlanLock` content-addresses configuration, logical/execution plans, model/tokenizer, compiler,
  component artifacts, profiles, privacy contract, and workload.
- `Runtime`, `Session`, and `PreparedMaterial` are opaque native handles. They MUST NOT be cloned,
  pickled, placed in public plans, or reconstructed by copying files. Closing, cancellation,
  retry, and retirement preserve freshness and bind-before-expose rules.

Configuration construction performs no model download, device discovery, preparation, launch,
or network activity. Sequence is configure, resolve, validate, compile, prepare, execute, measure.

## Typed planning

Logical planning separates semantic, numeric, and protected meaning. An executable value records
shape, exact numeric domain and rounding, bound certificate, protection representation, holders,
epoch, layout, and state lifetime. `MaskedRing`, `ArithmeticLabel`, `BooleanLabel`, `AdditiveShare`,
and `Ciphertext` are not interchangeable. Every change uses a named conversion whose parties,
equations, error, leakage, and lifetime effects are validated.

Named component slots bind implementations into a graph; they are not sequential transformers.
Methods define protocol equations and assumptions. Kernels implement operations for devices.
Model adapters define architecture semantics. Preparation providers produce typed fresh material.
Transports move framed bytes without changing protocol semantics.

The semantic IR is model-family-neutral. Layer identity and persistent-state kinds are explicit;
compiler and method passes MUST select semantic operators and capabilities rather than parse adapter
node names, parameter paths, or model-family labels. Implemented adapters cover dense Qwen2 and
Qwen3, hybrid Qwen3.5, Phi-4 mini, and Gemma 4 text configurations. Their fused projections,
rotary variants, shared KV, recurrent matrices, and convolution state lower into the same IR.
Nemotron, Kimi, GLM, and future adapters extend typed operator vocabulary when required instead of
forking model-specific compiler pipelines.

## Profiles

Profiles are versioned, strict bundles of topology, operation coverage, numeric policy, permitted
representations, conversions, implementation eligibility, and evidence requirements.

- `baseline.masked_linear_cpu` preserves the client-heavy masked-linear integration target. It
  is not a thin-client claim and requires its recovered numeric lock before reproduction claims.
- `research.single_evaluator` targets minimal Client work, model-aware offline Preparation, and
  one online encoded evaluator. It fails until attention, normalization, nonlinearities,
  rescaling, state, selection, sampling, and feedback are covered.
- BFV/FHE, two-online-worker, and attested plaintext designs are explicit comparison profiles only.

Local co-location simulates roles but does not establish administrative non-collusion. Deployment
changes placement, not protocol, numeric, or privacy choices.

## Native ownership

Rust owns model-data scans, compiler passes, arithmetic and cryptographic kernels, material
allocation, transport, scheduling, clocks, metrics capture, and adversary instrumentation. Python
owns ergonomic immutable configuration and result presentation. Optimized execution crosses FFI
for complete regions or plans, never per scalar, gate, label, tensor row, or network frame.

Built-in Rust code MAY use internal Rust traits. Separately compiled providers MUST use the
versioned C ABI described by the component standard. Rust ownership does not by itself establish
constant-time behavior, cryptographic correctness, isolation, or proof refinement.

Public Python lifecycle and ownership rules are specified in [Python API](python-api.md). CLI role
orchestration and prepared-only behavior are specified in [CLI](cli.md). Repository, distribution,
and runtime-data placement are specified in [Package structure](package-structure.md).

## Evidence boundary

Plans identify required claims and evidence; they do not contain evidence conclusions. Source
claims, implementation coverage, measurements, assurance outcomes, and deployment assumptions are
separate typed records. Failed attacks do not prove privacy. Formal checks apply only to their
stated model and do not establish implementation refinement unless separately recorded.
