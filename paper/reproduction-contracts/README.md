# Ten-method reproduction contracts

The author approved a complete study of ten independent PLLM reimplementations.
The implementation must preserve each selected paper's method. A substantial
adaptation needs explicit approval before implementation.

The starting [coverage audit](../TEN-METHOD-COVERAGE.md) identifies six component
adaptations and no qualified complete-system reimplementation. These are starting
conditions, not benchmark results against the source systems.

## Contracts

| Contract | Methods |
| --- | --- |
| [Delegation](../methods/delegation-contracts.md) | Gupta–Katz–Miers; Maverick |
| [Shared computation](../methods/shared-computation-contracts.md) | SIGMA; FuseFSS; BumbleBee; PUMA |
| [Encrypted inference](../methods/encrypted-computation-contracts.md) | NEXUS; THOR; CacheMir; ROSETTA |

The [ten-source lock](../methods/source-lock.v1.json) pins the governing PDF
bytes and cache paths. The final BumbleBee PDF now has a verified persistent
cache copy; its older cache and historical records remain separate.

Independent reviews in this directory check the canonical contracts above. They
do not supply competing implementation specifications or approve adaptations.

Each contract must identify these items:

1. The exact paper version and source bytes, with algorithm and page references.
2. The original model, workload, numerical rules, and quality reference.
3. Every role, trust assumption, corruption model, and permitted disclosure.
4. Setup, protected operations, conversions, state, output, and token feedback.
5. Reusable PLLM components and missing complete-system contracts.
6. Differences between the paper, author code, and existing PLLM probes.
7. Proposed adaptations that require a decision.
8. A complete validation path and the first source-faithful executable slice.

Author implementations remain specification and oracle inputs. PLLM does not
build, import, link, vendor, or benchmark them as its implementation. Mature
general-purpose cryptographic libraries are separate dependencies; they do not
supply a paper's packing, scheduling, state, or protocol implementation.

## Implementation sequence

Complete working slices in this order. Reuse validated components across methods.

1. Freeze the source contract and obtain decisions for substantial adaptations.
2. Implement a bounded native primitive with an independent mathematical oracle.
3. Compose a complete operator, including conversion, setup, and output recovery.
4. Bind the operator to model-neutral semantic, numeric, and state contracts.
5. Run a complete generated-model request through separate role processes.
6. Compare a pinned original model against its specified clear numeric reference.
7. Validate continuation, cancellation, malformed input, replay, and retirement.
8. Measure the complete implementation through the common benchmark harness.
9. Freeze finalists and claims, then run independent confirmation cohorts.

A primitive or operator result cannot satisfy a later gate. Required one-use
material remains fresh, party-local, and charged throughout this sequence.
Source fidelity does not require reproducing a bug that violates the paper's
contract. Such corrections need an explicit discrepancy record and tests.

## Shared model and execution requirements

The approved common model is **Qwen2.5-0.5B-Instruct**, with Apple-M CPU-first
execution. The [decision record](../methods/DECISIONS.md) pins the source and
records the approved bounded Poulpy qualification. Its torus representation
does not replace any source-method profile without a separate mapping. Lattigo
adoption is not approved. Original-paper models and numerical references remain required.
A source-specific model result does not become a matched cross-method result
through a shared benchmark command.
Encoder-only execution does not receive an autoregressive decode rate.
An extension to a new model or state contract needs its own validation and label.

The current semantic decoder supports rotary-position gated architectures. It has
no compiled LayerNorm or learned absolute-position operator. Its existing
`gelu_tanh` operator has a BF16 contract. A classic LayerNorm/GELU decoder therefore
needs reusable semantic and import contracts before it can supply a common
reference for the original papers that use it. It is not the common-model choice.

All kernels, layouts, schedules, cryptographic parameters, and precision settings
must remain part of the recorded composition. An implementation in Rust does not
permit a change to the algorithm, arithmetic, trust assumptions, or numerical
domain without disclosure and the required decision.

## Measurement requirements

Follow the [framework study plan](../FRAMEWORK-REWRITE-PLAN.md). Freeze a new
comparison-contract version before the publication campaign; retain historical
contracts and reports unchanged.

Required results include prefill time and throughput, setup-inclusive first-token
time, N−1 decode throughput, client upload and download, and all-link traffic.
Report public setup, private preparation, prefill, decode, cleanup, and any
unclassified cost separately. Semantic EOS work remains decode work even when it
occurs after the final emitted text token.

Methods with final batched verification must retain their output-acceptance
barrier. GKM's tentative tokens do not become accepted text until that check
passes. Draining a buffered response after verification is not model decode
throughput. Report semantic decode timing, verification and accepted first-output
timing separately; leave N−1 text-event rates ineligible when they measure only
buffer release. Smaller verification windows require their own approved profile.

Memory results include the complete client and every non-client role. Retain role
instance counts, process identities, host placement, individual peaks,
simultaneous role and host aggregates, and the simultaneous all-non-client peak.
Keep sampled phase peaks distinct from process lifetime high-water measurements.
Host RAM, device memory, disk storage, admission estimates, and provisioned
capacity retain separate fields.

The client tiers remain 256 MiB, 2 GiB, and 16 GiB. The main comparison requires a
fixed provider budget. Per-party 100/40 Mbps limits alone do not fix aggregate
provider resources when role counts differ. Hardware, security, numerical,
quality, lifecycle, and workload classes determine which results can be ranked.

## Decisions

| Decision | State | Consequence |
| --- | --- | --- |
| Complete ten-method reimplementation study | Approved by the author | Existing PLLM configuration results cannot substitute for the study. |
| Substantial protocol, numeric, state, security, or execution adaptations | Approval required before implementation | Record the exact proposal and the author's decision. |
| Older GKM W8A8/ring32 and packed-DPF comparator proposals | Not approved as a source reproduction | Do not activate them under the GKM reproduction label. |
| Common model and development hardware | Qwen2.5-0.5B-Instruct; Apple-M CPU first | Preserve original-model validation; qualify common-Qwen numerical and state extensions separately. |
| Full-HE dependency | Bounded Poulpy torus qualification approved | Two refreshes, one CPU thread, and an 8 GiB admission ceiling; source-method mappings remain separate. |

Specific decisions belong beside the affected source contract. A recommendation
or a proposed component does not count as approval.
