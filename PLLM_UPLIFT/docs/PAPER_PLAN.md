# Paper uplift and publication evidence

Working title: **PLLM: Reproducible, Cost-Aware Compilation for Private Language-Model Inference**.

The intended systems claim is that selecting representations, conversion boundaries, model-derived numeric bounds and one-time-material schedules jointly improves the measured privacy-compatible inference frontier. This is a hypothesis to test. Multi-protocol compilation already has precedent, including HyCC; native MPC frameworks and arithmetic garbling also predate PLLM. Cite them as foundations.

## Research questions

1. Can region-level selection improve over the best uniform backend under identical security, numeric and topology constraints?
2. How do setup, client work, conversions and sustained material supply change rankings based only on online kernels?
3. When do public certified bounds remove encoded arithmetic/storage without changing the declared function?
4. Which source methods survive a faithful native implementation and complete-operator accounting?
5. Can the preferred single-evaluator topology achieve competitive useful TPS and total cost on a real model?
6. Does the assurance layer discover real integration errors or rejected unsafe optimizations without overstating proof coverage?

## Minimum comparison design

Use original model runtime, exact clear numeric graph, same partition without privacy where meaningful, existing PLLM, fixed native method families and searched composition. Reproduce external original artifacts in isolated environments when available. External paper tables are source-reported context, never directly pooled with local measurements.

Maintain separate cohorts for two online servers, client-heavy methods, HE, TEE and changed quantization/approximations. Every table states corruption model, parameter profile, setup, leakage, hardware, precision, context, batch, request count, useful token count and client boundary.

## Ablations

Uniform versus regional selection; conversion-blind versus conversion-aware choice; box versus geometric bounds; unfused versus fused gates; flat/Boolean/logrow/weighted nonlinear implementations; public packed weights versus expanded storage; frozen versus sustainable inventory; independent per-op versus complete-block scheduling; attack checks disabled versus enabled overhead. These are experimental configurations, not permission to release an unchecked variant.

## Artifact structure

For each published figure, commit an experiment recipe, source/model locks, raw public samples, failed configurations, environment, compiler output, assurance contract and deterministic plotting/table script. Cite method sources automatically from the executed plan. Include numerical-quality and privacy-game definitions next to performance. Publish no live secret material.

A source paper's proof, a solver-checked finite property, a known attack witness and an empirical distinguisher result occupy distinct columns. Do not publish a passing test count as security bits. Our previous 10× nonlinear comparison is a reference component result with different baselines, not a full-model or across-the-board SOTA claim.

## Claim gates

“Reproduced” requires original source parameters/workloads, implementation fidelity notes and actual evidence. “Native speedup” requires native-to-native matched functionality. “Full-model” requires a real checkpoint with attention, private state, sampling and token feedback. “Near-native TPS” requires the same generation workload, compatible quality and all-party cost disclosure; a large frozen pool cannot conceal impossible replenishment.

Novel weighted transducers, mask-side-information codecs and range-aware compositions stay candidate contributions until an applicable proof/review and strong baselines exist. A well-executed negative result or discovered parameter/privacy flaw is useful research and belongs in the artifact rather than being silently removed.
