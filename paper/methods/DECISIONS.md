# First implementation decisions

Status: author decisions recorded. CPU-first Apple-Silicon execution and the
common Qwen2.5-0.5B model are approved. Bounded Poulpy qualification is approved;
backend adoption for the source methods remains open. D6–D9 approve named PUMA
and FuseFSS source-derived profiles, retain BumbleBee's final-paper target, and
select a fixed-count private token-feedback contract for the four shared-computation
methods. These decisions do not qualify complete implementations or measurements.

The author has approved genuine independent reimplementations of all ten methods.
Their original protocols, arithmetic, role ownership and security requirements are
the default. Secure randomness, defined arithmetic and rejection of invalid
results are mandatory. They do not need a separate adaptation decision.

The canonical [contracts](README.md) and independent
[audits](../reproduction-contracts/README.md) separate unresolved source facts
from the following actual changes.

## D1. CPU-first execution

**Approved:** first implement and validate complete CPU configurations on Apple
M-series hardware. Use ARM64 SIMD and native CPU acceleration where exactness,
source security requirements and complete costs permit. Retain
the original CPU algorithm when a source supplies one. Label CPU ports of
GPU-evaluated methods explicitly. Implement the method's protected operations,
conversions and state; do not move them into clear client or provider kernels.

The first complete path is Gupta–Katz–Miers with two workers, dense private
float32-row lookup, original fixed-point linear delegation, secp256k1 checks,
and request-end verified acceptance. Native scalar/oracle work precedes compiler
and model integration. SIGMA's CPU branch keeps its own source schedule.

**Why needed:** the available development host is an Apple M5 with 32 GiB RAM.
It is not the CUDA hardware used by several source evaluations. CPU correctness
and measurements must not inherit the papers' GPU performance.

**Validation:** independent arithmetic and protocol oracles; original-model
correctness and quality; complete role/lifecycle checks; explicit CPU, memory,
network and storage accounting. GPU configurations require separate backend and
hardware qualification. The final fixed-provider-budget campaign is not selected
by this development decision.

## D2. Full-HE primitive dependency qualification

**Author direction:** prefer an existing native Rust backend. First examine its
actual primitives, source parameters and Apple-Silicon CPU support. A Rust wrapper
around a Go/C++ library is not a native Rust backend.

The earlier Lattigo v6.1.1 proposal was not approved. Its pinned candidate,
`37a34b4676e987795949a795f0ce9d0893763c96`, remains comparison material. PLLM
independently implements source-specific packing, lookup, scheduling, state and
role protocols regardless of primitive-library choice.

**Why needed:** the installed SEAL path has no real CKKS refresh or the required
cross-scheme conversion. Lattigo's lower-level encrypted-accumulator blind rotation
is a candidate primitive; its public-LUT wrapper is not a ROSETTA replacement.

**Validation before adoption:** actual source parameter/key graphs, client-only
secret-key custody, real refresh, required coefficient/slot and ring transforms,
encrypted-accumulator selection, target-level recovery, independent numerical
oracles, resource admission and complete primitive costs. A capability failure
requires another explicit proposal rather than a weaker protocol.

The [Rust backend review](rust-he-backend-qualification.md) identifies Poulpy's
existing CKKS refresh and Apple-M CPU primitives. D4 authorizes a bounded test of
that candidate. It does not establish that the library implements a paper.
Present specific missing capabilities and alternatives before adding a backend
in another language. NEXUS/THOR numerical differences remain separate
source-resolution decisions.

## D3. Common Qwen2.5-0.5B comparison track

**Approved:** retain original-model validation for each method, then add a named
common Qwen2.5-0.5B-Instruct track. Continue the existing research source pin:

- Repository: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Pin source: [research/common.py](../../benchmarks/research/common.py).
- The proposed common GPT-2 track is not selected. GPT-2 source-specific checks
  retain their own 1024-position bound where applicable.

Verify the existing complete checkpoint/tokenizer lock before import and freeze
full workload/quality rules before comparison. The common model choice does not
approve PLLM W8A8 as a substitute for each method's source numerical profile.

**Why needed:** the papers do not all evaluate the same model or functionality.
THOR evaluates BERT encoders; NEXUS and Maverick do not establish the complete
persistent generation lifecycle required by the proposed study. Their common
decoder configurations are extensions, not original evaluated workloads.

**Preserve:** each method's numerical rules, cryptography, roles, weight privacy,
output disclosure and permitted client operations. Compare exact correctness to
each method's own clear oracle, then apply a frozen quality criterion before cost
ranking. Do not force all methods into PLLM W8A8/ring32.

**Additional decisions remain necessary:** unresolved approximation profiles and
any new persistent KV policy. D6–D9 now settle the specified PUMA/FuseFSS numeric
choices and shared-computation token-feedback/output contract; other method
contracts remain separate.
Use full-prefix feedback where that is the evidenced original schedule; do not
silently replace it with a faster cache. Existing encrypted-cache methods retain
their own layouts and state rules. This model decision does not approve arbitrary
boundary or state substitutions.

## D4. Bounded Poulpy backend qualification

**Approved:** qualify native-Rust Poulpy v0.9.0 at commit
`ca3532570ff62b96599a74f7adb85de7726b0446` with its power-of-two torus
representation. Use a distinct backend-qualification profile on Apple-M CPU.
The approved limits are at most two encrypted refreshes, one CPU execution
thread, independent numerical checks, and an 8 GiB admission ceiling.

Freeze the dependency and toolchain, parameter/key graph, bounded fixture,
precision criteria, operations, and resource estimates before execution.
Price keys, contexts, setup, copies, and scratch together. Retain the host
reserve, monitor swap growth, and stop if admission fails. Build and setup have
separate bounded budgets. Keep this qualification outside the stable production
workspace until its toolchain and dependency requirements are qualified.

The evaluator must use public evaluation material without a secret key or
client refresh. Check every declared slot and report width, scale, layout,
error, and resource observations. A failed test does not authorize changed
thresholds or a weaker primitive.

**Approval boundary:** this decision permits the bounded torus qualification.
It does not approve replacement of NEXUS, THOR, CacheMir, or ROSETTA's source
parameters, representations, approximation schedules, or conversion protocols.
Each method still needs an explicit source mapping. TFHE-rs, OpenFHE, and
Lattigo adoption are separate decisions. No model-performance result follows
from a successful refresh test.

**First gate result:** the pinned eager `NTT4x30Portable` key bundle rejects
before allocation. Its 33 prepared automorphism keys alone need 29.78 GiB.
Borrowed preparation also keeps raw keys live, for a 37.22 GiB overlap before
other owners. No encrypted refresh or native build completed. The
[admission contract](../../benchmarks/native/poulpy_refresh/README.md) preserves
the rejected attempts and independent arithmetic. A bounded key-residency design
must retain the approved parameters and price every live owner before execution.
The 8 GiB cap remains unchanged.

The separate [staged owner review](poulpy-staged-owner-bound.md) also rejects
the proposed five-wide-ciphertext, seven-key policy when it uses the direct
public phase scratch query. Its charged subset is 8,613,846,336 bytes, already
23,911,744 bytes above the cap. The matrix-aware query has a smaller lower
bound, but no complete upper bound: allocator retention, private preparation
buffers, typed cross-role storage, and coefficient-solver setup remain unresolved.
These source calculations authorize no encrypted execution. Parameter security
and second-refresh numerical admission remain separate gates.

## D5. Ordinary SDK integration is required

**Author direction:** every method must fit the existing SDK capability contracts
so users can compose it through `Pipeline`, `Experiment`, the CLI and benchmarks.
Native or internal research execution is an implementation milestone, not a
completed method.

Define immutable component configuration and its numerical, role, verification,
state and lifecycle contracts before expanding a method-specific implementation.
Reuse component discovery, serialization, source resolution, native compilation,
role supervision, client/gateway execution and benchmark evidence. Extend a
shared contract when necessary; do not create an independent public runner,
configuration schema or method-named CLI branch.

**Required validation:** component discovery and exact round trips; Pipeline and
Experiment resolution; early rejection of unsupported compositions; generated
complete requests through the ordinary SDK and local gateway; the exact published
CLI invocation; and ordinary benchmark/replay with complete ownership accounting.
Request cancellation, partial output, malformed replies and final acceptance must
preserve each method's source contract across those entry points.

Operator-only availability and complete-response availability remain separate
capabilities. An installed operator does not qualify an unsupported decoder.
Internal references must retain their scope until their public execution path
passes these gates. The subsequent bounded GKM SDK slice now has genuine
party-local workers and verified request integration. Its generated-checkpoint
conformance does not qualify full-checkpoint execution or performance; the earlier
in-process reference retains its separate scope.
The [SDK integration contract](sdk-integration.md) records the shared extension
points and concrete end-to-end completion checks.

## D6. Approved PUMA source-derived Q18 profile

**Approved 17 October 2026 via the user question tool:** adopt the named profile
`puma.source-derived-q18.v1`, exactly as specified in
[source resolution §3.3](../reproduction-contracts/shared-profile-resolution.md#33-approved-p-n1-profile-source-derived-q18).
This selects the documented source-derived interpretation, not bit identity with
the missing evaluated graph.

The approved rules include ring64/Q18, finite binary32/toward-zero encoding,
pinned replicated arithmetic and stochastic `TruncAPr`; the stated GELU/SiLU
constants and evaluation order; shifted clipping before summation with five
softmax squares and encoded-zero exponential epsilon; and centered mean-square
LayerNorm with encoded epsilon two and the documented reciprocal/rsqrt schedules.
The explicit Llama RMSNorm/RoPE and checkpoint/frontend gaps in §3.3 remain gaps.

**Effect:** P-N1's adoption approval gate is closed for this named profile.
Complete operator/graph implementation, source-model/checkpoint locks, per-product
range proofs, mixed-kernel PRSS scheduling, protected transport, D5 SDK integration
and quality/cost qualification remain open. Original evaluated-table reproduction
still requires its missing graph/configuration evidence.

## D7. BumbleBee final-paper target

**Approved 17 October 2026 via the user question tool:** retain the NDSS final
paper as the governing target. Do **not** adopt the artifact's Q16/four-squaring profile or
combine artifact settings with final-paper settings by convenience.

**Effect:** target choice is settled; **B-N1 final-paper mapping remains open**.
Obtain the evaluated numeric/configuration/HE manifest enumerated in
[source resolution §4.4](../reproduction-contracts/shared-profile-resolution.md#44-b-n1-final-paper-mapping-remains-open):
rounding, per-node rings/scales, activation constants/endpoints, normalization,
truncation/error flags, ordered moduli/levels, noise/key switching and frontend
identity. Candidate artifact constants/primes remain evidence, not adopted
final-paper parameters. D9 selects the new transaction's output policy without
establishing the original evaluated frontend identity.
Independent HE/RLWE, noise, key-graph and circuit-privacy qualification also
remain open; candidate prime lists do not establish their security.

## D8. Approved FuseFSS source-derived descriptor profile

**Approved 17 October 2026 via the user question tool:** adopt the named profile
`fusefss.source-derived-ring50-q12.v1`: ring50/Q12 descriptors, nearest-even
**external input and weight** encoding, and SIGMA's floor-defined CPU tables,
ARS/rescales and internal evaluation order, as specified in
[source resolution §5.1](../reproduction-contracts/shared-profile-resolution.md#51-approved-f-n1-descriptor-profile).
External nearest-even encoding is not permission to change internal floor rules.

**Effect:** F-N1's adoption approval gate is closed for this named profile.
Freeze and implement every gate descriptor/LUT/coefficient/rescale/Phi byte,
secure backend and complete graph; establish range proofs, D9 boundaries,
transport, D5 SDK integration and quality/cost evidence. The missing evaluated
descriptor artifact still blocks claims of original FuseFSS numerical identity.

## D9. Fixed-count private token feedback

**Approved 17 October 2026 via the user question tool:** PUMA, BumbleBee, SIGMA
and FuseFSS use a public fixed generation count, protected full-prefix recomputation for every
selection, lowest-index protected argmax, private token-ID feedback/append,
and final **client-only token-ID reveal** after the complete loop. Equal encoded
maximum scores select the lowest vocabulary index. There is no intermediate
token/logit reveal or EOS-dependent early exit in this contract.

For SIGMA/FuseFSS, explicitly add and charge the missing protected token embedding,
final normalization and complete vocabulary head. The selected completion uses
protected equality→one-hot embedding and preserves each method's model ownership,
online roles and dealer assumptions. See the
[complete transaction](../reproduction-contracts/shared-profile-resolution.md#6-token-feedback-and-state-precise-evidenced-boundary).

**Effect:** T-N1's boundary/tie approval gate is closed. Full protected execution,
fresh per-step material, cancellation/retirement, prefill plus multiple actual
feedback selections, admission, D5 integration and quality/cost evidence remain
required. Persistent KV, streaming disclosure, stochastic sampling, full logits,
EOS-short-circuit and cross-request continuation are not approved by D9.

## Decision record

| ID | State | Blocks |
| --- | --- | --- |
| D1 | Approved: Apple-M CPU first | Source-faithful native development may proceed; source GPU timing claims remain separate. |
| D2 | Prefer an existing Rust backend | Poulpy is the selected qualification candidate under D4; source-method adoption remains open. |
| D3 | Approved: Qwen2.5-0.5B-Instruct | Common-model work preserves each method's numerics; substantial protocol/state resolutions still need approval. |
| D4 | Approved: bounded Poulpy torus qualification | At most two refreshes, one CPU execution thread, independent numerical checks, and an 8 GiB admission ceiling; no implied source-method mapping. |
| D5 | Required: ordinary SDK integration | Component, Pipeline, Experiment, compiler, SDK/gateway, CLI and benchmark checks gate complete-method status. |
| D6 | Approved: `puma.source-derived-q18.v1` | P-N1 adoption closed for §3.3's named interpretation; implementation/range/transport/SDK/quality and evaluated-build identity remain separate. |
| D7 | Approved: BumbleBee final-paper target | Artifact Q16 is not adopted; B-N1 evaluated-final-paper mapping remains open. |
| D8 | Approved: `fusefss.source-derived-ring50-q12.v1` | F-N1 adoption closed; nearest-even external encoding and SIGMA floor internals are explicit; full descriptors/backend/graph qualification remains open. |
| D9 | Approved: fixed-count private full-prefix feedback | T-N1 choice closed; lowest-index argmax, final client-only IDs, and charged SIGMA/FuseFSS boundary completion still need implementation and qualification. |

D6/D8 settle only their explicitly named source-derived interpretations. BumbleBee
final-paper mapping, NEXUS numerical conflicts and any further adaptation still
need concrete source evidence or their own decision. This register grants no
blanket approval and changes no measured history or complete-method status.
