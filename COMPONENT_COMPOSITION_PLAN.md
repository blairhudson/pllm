# Component composition migration

This document is the working implementation plan for replacing profile-driven
execution with component-driven pipeline composition. It also records the
correct classification of MPCache-derived work. Update this document as gates
close; do not use profile, paper, or model-family names as compiler dispatch.

## Decisions

1. A `Pipeline` composition is canonical. Its behavior comes from its model and
   resolved components, not from a profile name or Python subclass.
2. Classes under `pllm.profiles` are optional convenience constructors. A
   manually composed pipeline and a convenience-constructed pipeline with the
   same model and components have identical canonical bytes and digest.
3. Preset names, when retained for display, are non-semantic metadata. They do
   not select compiler, runtime, coverage, or evidence behavior.
4. `research.single_evaluator` is removed. "Single evaluator" describes a role
   topology that may result from resolving component requirements; it is not a
   profile or method.
5. The resolver derives roles, trust assumptions, representation transitions,
   material lifetimes, operator coverage, and execution requirements from
   component contracts.
6. MPCache is a paper and provenance source for cache optimizations, not a model
   family or runtime dispatch identity.
7. Existing versioned benchmark evidence may retain historical profile labels.
   New runtime configuration does not keep compatibility aliases for profile
   dispatch.

## Target configuration

Generic composition must be sufficient:

```python
pipeline = Pipeline(
    model=model,
    components={
        "linear": MaskedLinear(),
        "preparation": ModelAwareCorrections(),
        "nonlinear": R03CrtGatedMultiplyQ7(),
        "schedule": ChunkedIndependentLanes(),
        "kernels": Cpu(),
    },
)
```

A convenience constructor such as `MaskedLinearCpu(model)` may produce that
composition, but neither the serializer nor the resolver depends on its class.

## Phase 1: canonical composition

- Remove the required `profile` field from `Pipeline` and experiment schema.
- Canonicalize model plus components only.
- Introduce a resolved-pipeline record containing the normalized component
  graph, pipeline digest, derived roles, trust contract, and runtime contract.
- Rename `ExperimentProfile` to a composition-neutral resolved-experiment type.
- Group new benchmark evidence by pipeline digest and exact component identity.
- Support historical profile fields only in versioned evidence readers where
  persisted records require them.
- Prove that manual and convenience construction serialize and hash identically.

## Phase 2: component contracts and resolution

Each component contract declares:

- category and provided capabilities;
- covered semantic operators;
- consumed and produced representations;
- numeric domain and conversion requirements;
- required roles and communication edges;
- trust, corruption, and non-collusion assumptions;
- preparation and one-use material lifecycle;
- host and kernel requirements;
- state and scheduling compatibility;
- assurance and evidence gates.

The resolver merges compatible declarations into one normalized role topology
and rejects gaps or conflicts before native compilation. It does not inspect
profile classes or paper/model names.

## Phase 3: compiler and runtime migration

- Remove `SILU_Q7_EXPERIMENT_PROFILE` and all profile-name compiler branches.
- Replace `ModelPlan.coverage(profile: str)` with coverage of a resolved
  component composition.
- Replace schedule and compile-request profile arguments with the resolved
  pipeline identity and component graph.
- Bind executable regions to component IDs, versions, parameters, artifact
  digests, representations, and material contracts.
- Derive runtime options from resolved capabilities instead of `isinstance`
  checks against convenience profile classes.
- Make `Pipeline.from_spec()` sufficient for every supported composition.
- Keep existing convenience profiles covered by parity and end-to-end tests.

## Phase 4: protected execution composition

Replace the former `research.single_evaluator` target with ordinary components:

| Concern | Component family |
| --- | --- |
| Masked or delegated linear work | Protocol method |
| Offline material | Preparation provider |
| Garbled nonlinear work | Nonlinear protocol |
| Label encoding | Label scheme |
| Q14, Q10, and Q7 transitions | Conversion or numeric transform |
| Chunking and lane execution | Protected scheduler |
| KV ownership and lifetime | State protocol |
| Result checking | Verification scheme |
| Native execution | Kernel backend |
| Processes and communication | Derived role topology |

Complete protected decoder execution only when one resolved composition covers
every semantic operator without plaintext or undeclared fallback.

## Phase 5: MPCache decomposition

Keep paper identity in R23 provenance. Use capability-based implementation IDs:

| Capability | Component family |
| --- | --- |
| Bounded cache-eviction graph rewrite | `KvCacheEviction` compiler pass |
| Cache importance scoring | Semantic or protected operator |
| Protected selection | `ProtectedTopK` nonlinear/protocol component |
| Protected selected-state movement | `SelectedKvGather` conversion/data movement |
| Attention over selected state | `SelectedKvAttention` execution component |
| Full or shared KV ownership | `ClientLocalKv` or `SecretSharedKv` state protocol |
| Material and message ordering | Scheduler/protocol component |

Remove paper-branded runtime IDs such as `pllm/mpcache/v1`. Model adapters expose
cache and state semantics only. Qwen, Gemma, Phi, and future adapters remain
separate model-support tracks.

## Phase 6: research and status truth

- Add canonical component, method, source, SDK, code, test, and example links for
  R01 through R24.
- Generate implementation status from lifecycle and evidence records, not paper
  ID switches.
- Correct stale R03, R04, R07, R19, and R23 claims.
- Distinguish selectors, structural transforms, executable operators, and
  end-to-end examples.
- Require reciprocal SDK and research links and execute every runnable example.

## Subsequent vertical slices

1. Public compiled-region execution.
2. Representation and conversion contracts.
3. Fully composed protected Qwen execution.
4. Real-checkpoint dense Qwen3 evidence.
5. Gemma local operators and checkpoint execution.
6. Phi LongRoPE execution.
7. Qwen3.5 recurrent and convolution state.
8. Protected cache-eviction components derived from R23.
9. Native plugin loading and remote deployment.
10. Canonical benchmark, comparison, and search CLI.
11. GPU and modality support.
12. R01 through R24 clean-room reimplementations in dependency order.

Each slice requires a source/provenance lock, bounded reference implementation,
differential tests, native integration where applicable, end-to-end execution,
matched benchmark evidence, documentation, and an evidence-derived status update.

## Acceptance gates

- No compiler or runtime branch dispatches on profile, paper, or model-family
  names.
- Generic and convenience construction produce the same pipeline, plan, and
  execution behavior.
- Incompatible representation, topology, material, state, or trust contracts
  fail before execution.
- Existing gateway and benchmark workflows resolve through generic composition.
- Rust tests, Clippy, Python tests, documentation checks, runnable examples,
  distribution checks, and the production documentation build pass.
- Published support claims identify the exact component composition, model
  fingerprint, workload, evidence cohort, and limitations.
