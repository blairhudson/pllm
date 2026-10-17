# Ordinary SDK integration contract

Status: required by [D5](DECISIONS.md#d5-ordinary-sdk-integration-is-required).
This document records integration requirements and the bounded GKM milestone.
The requirements apply to every selected method and its separately labelled adaptations.

## One public execution path

Each method must use the existing chain:

```text
typed capability components
    -> canonical Pipeline and Experiment
    -> source/workload-aware native admission
    -> compiled model and role execution
    -> SDK / gateway / CLI / ordinary benchmarks
```

Component descriptors, reconstruction and serialization use `ComponentRef`,
`ComponentDescriptor` and `create_component`. Register genuine capabilities in
their existing families; do not add a parallel method configuration system.
Seeds, keys, pending operations, publication leases and provider credentials stay
in runtime owners and never become component parameters.

The compiled numerical, state, role and acceptance contracts govern every entry
point. Unsupported combinations reject explicitly before graph expansion, tensor
loading, role launch or secret issuance, as appropriate to the rejected input.
No entry point may choose a different method to make the configuration execute.
Publication or lifecycle work performed locally remains charged to its real owner.

Operator benchmarks are useful intermediate capabilities. Component registration,
configuration round trips and operator execution do not establish complete SDK
response support. A native or internal reference remains a reference until its
ordinary public execution path passes the required tests.

## Reusable paper configurations and measurements

The study's executable entry points live in `examples/papers/`: one small Python
Experiment factory per implemented method, a shared model/budget, and one input
file. Run them with `pllm benchmark run` and `pllm benchmark quality`. Missing
execution or measurement capabilities belong in the shared SDK. Method-specific
benchmark runners, metric implementations and one-use archive tools are not the
implementation path. Existing archived measurements retain their original scope.

GKM is the first paper factory. Its quality adapter uses the installed worker
transport and native acceptance, then the existing common float32 reference and
agreement metrics. Same-token decode is a quality-driver input; ordinary SDK
responses retain fixed-count greedy generation. The remaining nine factories
arrive with their executable SDK compositions.

## Bounded GKM milestone and remaining gates

The [bounded decoder](../../docs/research/gkm-decoder-reference.md) reuses source
lowering, compiler matrix grouping and `SemanticLocalOperators`. Its repaired
native and installed-wheel checks validate that in-process composition.
`GkmDecoderSession` continues to supply that in-process control. The subsequent
`OpenWeightVerifiableCpu` profile composes `OpenWeightVerifiableOutsourcing`,
`PublishedLinearVerify`, `TwoOnlineDelegationRoles`, one-thread CPU kernels and
`ClientLocalKv` through the ordinary public execution path.

The bounded slice executes a fixed-length greedy request through one serialized
Experiment and two actual worker adapters. It retains a complete tokenizer and
vocabulary, original float32 lookup, source i16/Q11 encoding, ring64 products,
complete head, full client KV and request-end verification. Generated Qwen2,
Qwen3 and Llama fixtures fit the vocabulary ceiling of 256; default
`Model.tiny()` retains 258 entries and rejects. Co-located workers establish
functionality, not independent operators. Exact contracts and the 547 Python /
117 native conformance checks are in the
[SDK reference](../../docs/research/gkm-sdk-reference.md) and
[archive](../artifacts/gkm-sdk/README.md).
The subsequent [streaming-import slice](../artifacts/gkm-sdk/source-stream.md)
retains the same ordinary path and passes 571 installed-wheel and 120 native
checks. Bounded original-value frames replace eager Python source buffers;
encoded weights move into their final worker owners. Full-width admission and
whole-client memory measurements remain separate gates.

The [explicit Decoder-capacity slice](../artifacts/gkm-sdk/decoder-capacity.md)
then passes 640 installed Python and 134 native checks, including complete
generated two-worker results beyond the Capsule vocabulary ceiling. Native source
geometry and owner bounds admit pinned Qwen metadata. Its source-present
original preflight is host-admission blocked: 10.83 GiB required versus 8.11 GiB
safe headroom. A later [source-present Qwen response](../artifacts/gkm-sdk/qwen-response.md)
passes fresh admission and accepts four input tokens plus two outputs through both
workers. Full-width source import, publication and checked runtime arithmetic
complete for that workload. Independent full-logit/KV and quality evidence remain
open.

Broader activation coverage, original-model and common-Qwen
quality and matched resource/performance evidence remain open. Seeded network
frames, stochastic or early-EOS generation, continuation and automatic network
planning remain unsupported. Hub checkpoint resolution now uses the shared SDK
resolver while preserving the original model specification and source lock.

### Integration seams

| Shared contract | Implemented bounded GKM binding |
| --- | --- |
| `protocols`, `verification`, `roles`, component discovery and exports | Typed declarations with exact parameters and immutable round trips. Keep private material out of configuration. |
| `profiles.resolve_runtime_composition`; native composition classification and resolution | Select explicit source arithmetic and acceptance. Existing W8A8 defaults cannot supply GKM numerics. |
| `GkmDecoderContract`, `modeling.lower_model`, benchmark memory admission | Run source-mapped budget checks before any graph construction, including estimation and CLI inspection paths. Bind source, tokenizer, workload and canonical composition identities. |
| Native GKM window and publication owners | Separate client issuance/verification from worker evaluation. Each worker receives only its role's shares and permitted seed material. Bound wire frames by publication, request, stage, ordinal, worker, shape and profile. |
| `roles.topology`, `runtime.servers`, `runtime.party`, deployment and native placement | Dispatch from the admitted method capability. Shared `worker_a`/`worker_b` names alone cannot select GKM; the explicit delegated topology selects its worker adapter. Automatic network placement remains unsupported. |
| `runtime.model_binding`, `RuntimeClient`, transport and sidecar | Reuse semantic execution and source/tokenizer validation; add the admitted numerical boundary and verified-response lifecycle. |
| CLI, `BenchmarkTarget`, `benchmark_suite` and coverage | Consume the same Experiment; price actual owners and preserve numerical, acceptance, phase and traffic identities through replay. Automatic search remains unsupported. |

### Acceptance and request semantics

Tentative values may drive private client execution but cannot become accepted
response content, streaming token deltas, completion events or reusable state
before the final native check succeeds. Failure or cancellation retires pending
client and worker operations and drains outstanding calls.

The existing reference uses fixed-length greedy execution; EOS does not shorten
its window. The first public composition must identify and enforce that policy.
Ordinary sampling defaults must not silently substitute another policy. Early
termination, stochastic sampling, continuation and seeded transport require their
own admitted contracts and tests before being advertised.

Benchmark first accepted output includes final verification. Releasing buffered
tokens afterward is not model decode throughput. Preserve semantic decode and
verification timings separately; mark N−1 text-event decode ineligible when it
only measures buffer release. Source publication, import, provider work, client
state, framing and retirement stay visible under their actual owners and windows.

## Completion evidence

1. Discover and reconstruct each typed component; round-trip its Pipeline and
   Experiment with equal digests. Supported parameter changes alter identity.
2. Resolve the composition natively and reject incompatible numerical, verifier,
   state, kernel and topology choices before resource ownership begins.
3. Exercise oversized-source rejection through SDK, CLI and memory estimation,
   proving those paths cannot bypass pre-lowering admission.
4. Execute real role exchanges with independent party-local state. Test binding
   substitution, replay, truncated frames, malformed replies and cancellation.
5. Compare original lookup bytes, complete logits, active KV, history and pending
   selections to the independent oracle across supported source adapters.
6. Inject final verification failure through SDK and gateway. Confirm that no
   model-derived output or reusable state escapes the acceptance barrier.
7. Execute the published CLI example and an ordinary complete-response benchmark.
   Replay must bind the same Experiment, source, numerics and acceptance policy.
8. Update existing benchmark capability coverage and public API inventories with
   measured, unsupported and blocked cells kept distinct.

The same requirements govern later methods. Original-model validation, current
cryptographic review and matched publication measurements remain separate gates.
